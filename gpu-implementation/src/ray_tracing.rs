use cudarc::curand::CudaRng;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, DriverError, LaunchConfig,
    Profiler, PushKernelArg,
};
use cudarc::nvrtc::CompileError;
use log::error;
use std::any::Any;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

pub mod cuda_types {
    use cudarc::driver::DeviceRepr;

    pub type FloatSize = f32;

    #[repr(C)]
    struct Pixel {
        r: FloatSize,
        g: FloatSize,
        b: FloatSize,
        a: FloatSize,
    }
    // #[repr(C)]
    // pub struct Camera {
    //     pub aspect_ratio: FloatSize,
    //     pub vfov: FloatSize,
    //     pub image_width: u32,
    //     pub image_height: u32,
    //     pub center: [FloatSize; 3],
    //     pub pixel00_loc: [FloatSize; 3],
    //     pub pixel_delta: [FloatSize; 3],
    //     //
    //     pub origin: [FloatSize; 3],
    //     pub look_at: [FloatSize; 3],
    //     pub v_up: [FloatSize; 3],
    //
    //     pub u: [FloatSize; 3],
    //     pub v: [FloatSize; 3],
    //     pub w: [FloatSize; 3],
    // }

    #[repr(C)]
    #[derive(PartialEq)]
    pub struct DeviceGUI {
        pub sample2_per_pixel: u32,
        pub block_dim: u32,
        pub max_depth: u32,
        pub show_random: bool,
        pub random_norm: bool,
    }

    impl Default for DeviceGUI {
        fn default() -> Self {
            Self {
                show_random: false,
                random_norm: false,
                sample2_per_pixel: 4,
                block_dim: 1024,
                max_depth: 4,
            }
        }
    }
    // unsafe impl DeviceRepr for Camera {}
    unsafe impl DeviceRepr for DeviceGUI {}
}

use crate::gui::{DebugGui, Gui};
// use cuda_types::Camera;
use camera::Camera;
use cuda_types::FloatSize;

pub struct CudaWorld {
    ctx: Arc<CudaContext>,
    render: Option<CudaFunction>,
    d_frame: CudaSlice<u8>,
    rng_block: CudaSlice<u32>,
    rng: CudaRng,
    gui: Rc<RwLock<DebugGui>>,
}

// const PTX_SRC: &str = concat!(include_str!("cuda/floatN_helper.cu"), include_str!("cuda/lib.cu"), include_str!("cuda/kernel.cu"));
const PTX_SRC: &str = concat!(
    include_str!("cuda/floatN_helper.cu"),
    include_str!("cuda/library.cu"),
    include_str!("cuda/ray.cu")
);

use std::fs;
use std::io::Write;

fn load_ptx_src() -> std::io::Result<String> {
    let floatn = fs::read_to_string("src/cuda/floatN_helper.cu")?;
    let library = fs::read_to_string("src/cuda/library.cu")?;
    let ray = fs::read_to_string("src/cuda/ray.cu")?;
    Ok(floatn + &library + &ray)
}

pub enum PtxError {
    CompileError,
    CompileErrorOther(CompileError),
    LoadFunction(DriverError),
}

impl CudaWorld {
    pub fn new(frame_size: usize, gui: Rc<RwLock<DebugGui>>) -> Self {
        let ctx = CudaContext::new(0).expect("Failed to create CudaContext");
        let stream = ctx.default_stream();
        let rng = CudaRng::new(0, stream.clone()).expect("Failed to create CudaRng");
        let mut rng_block = stream
            .alloc_zeros::<u32>(frame_size)
            .expect("Failed to allocate frame buffer on device");
        rng.fill_with_uniform(&mut rng_block)
            .expect("Failed to fill rng block");
        let d_frame = stream
            .alloc_zeros::<u8>(frame_size)
            .expect("Failed to allocate frame buffer on device");

        Self {
            ctx,
            render: None,
            d_frame,
            rng_block,
            rng,
            gui,
        }
    }

    pub fn compile_ptx(&mut self) {
        if self.gui.read().unwrap().compile_ptx {
            let ptx_scr = load_ptx_src().expect("Failed to load PTX src");
            let ptx_compilation = match cudarc::nvrtc::compile_ptx(ptx_scr) {
                Err(CompileError::CompileError {
                    nvrtc: _,
                    options: _,
                    log,
                }) => {
                    error!("{}", log.as_c_str().to_str().unwrap());
                    Err(PtxError::CompileError)
                }
                Err(err) => Err(PtxError::CompileErrorOther(err)),
                Ok(ptx) => {
                    let module = self.ctx.load_module(ptx).expect("Failed to load PTX");
                    module
                        .load_function("render")
                        .map_err(PtxError::LoadFunction)
                }
            };
            match ptx_compilation {
                Ok(render) => {
                    self.render = Some(render);
                    self.gui.write().unwrap().compile_ptx = false;
                }
                Err(err) => {
                    let mut gui = self.gui.write().unwrap();
                    gui.render_msg = "PTX Compilation failed".into();
                    gui.compile_ptx = false;
                }
            };
        }
    }

    pub fn render(&mut self, frame: &mut [u8], camera: Rc<RwLock<Camera>>) {
        // check if ptx is compiled

        if self.render.is_none() {
            frame.iter_mut().enumerate().for_each(|(i, pixel)| {
                *pixel = (i % 255usize) as u8;
            });
            return;
        }

        let stream = self.ctx.default_stream();
        let mut binding = stream.launch_builder(self.render.as_ref().unwrap());
        if self.gui.read().expect("Gui can't be read").random {
            self.rng
                .fill_with_uniform(&mut self.rng_block)
                .expect("Failed to fill rng block");
            self.gui.write().expect("Gui can't be written").random = false;
        };
        camera.write().unwrap().initialize();
        let launch = {
            let gui = self.gui.read().expect("Gui can't be read");
            let camera = camera.read().expect("Camera can't be read");
            let builder = binding
                .arg(&self.rng_block)
                .arg(&mut self.d_frame)
                .arg(&*camera)
                .arg(&gui.device_gui);
            let dim = gui.device_gui.block_dim;
            let launch_cfg = LaunchConfig {
                block_dim: (dim, 1, 1),
                grid_dim: (
                    (camera.image_width * camera.image_height).div_ceil(gui.device_gui.block_dim),
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(launch_cfg) }
        };
        if let Err(err) = launch {
            self.gui.write().unwrap().render_msg =
                err.error_name().unwrap().to_str().unwrap().into();
        } else {
            stream
                .memcpy_dtoh(&self.d_frame, frame)
                .expect("Failed to copy device frames");
        }
    }
}
pub mod camera {
    use crate::ray_tracing::cuda_types::FloatSize;
    use cudarc::driver::DeviceRepr;

    type Vec3 = [FloatSize; 3];

    fn dot(u: Vec3, v: Vec3) -> f32 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    }

    fn cross(u: Vec3, v: Vec3) -> Vec3 {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    }

    fn length(v: Vec3) -> f32 {
        dot(v, v).sqrt()
    }

    fn unit_vector(v: Vec3) -> Vec3 {
        let len = length(v);
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn subtract(a: Vec3, b: Vec3) -> Vec3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn add(a: Vec3, b: Vec3) -> Vec3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    fn scale(v: Vec3, s: f32) -> Vec3 {
        [v[0] * s, v[1] * s, v[2] * s]
    }

    #[repr(C)]
    #[derive(Debug, PartialEq)]
    pub struct Camera {
        pub aspect_ratio: f32,
        pub image_width: u32,
        samples_per_pixel: u32,
        max_depth: u32,
        pub vfov: f32,
        pub lookfrom: Vec3,
        pub lookat: Vec3,
        vup: Vec3,

        pub image_height: u32,
        pixel_samples_scale: f32,
        center: Vec3,
        pixel00_loc: Vec3,
        pixel_delta_u: Vec3,
        pixel_delta_v: Vec3,
        u: Vec3,
        v: Vec3,
        w: Vec3,
    }

    impl Camera {
        pub fn new(aspect_ratio: f32, image_width: u32) -> Self {
            let mut cam = Camera {
                aspect_ratio,
                image_width,
                samples_per_pixel: 10,
                max_depth: 10,
                vfov: 90.0,
                lookfrom: [-2.0, 2.0, 1.0],
                lookat: [0.0, 0.0, -1.0],
                vup: [0.0, 1.0, 0.0],
                image_height: 0,
                pixel_samples_scale: 0.0,
                center: [0.0; 3],
                pixel00_loc: [0.0; 3],
                pixel_delta_u: [0.0; 3],
                pixel_delta_v: [0.0; 3],
                u: [0.0; 3],
                v: [0.0; 3],
                w: [0.0; 3],
            };
            cam.initialize();
            dbg!(&cam);
            cam
        }

        pub fn initialize(&mut self) {
            self.image_height = (self.image_width as f32 / self.aspect_ratio) as u32;
            self.image_height = self.image_height.max(1);

            self.pixel_samples_scale = 1.0 / self.samples_per_pixel as f32;
            self.center = self.lookfrom;

            let focal_length = length(subtract(self.lookfrom, self.lookat));
            let theta = self.vfov.to_radians();
            let h = (theta / 2.0).tan();
            let viewport_height = 2.0 * h * focal_length;
            let viewport_width =
                viewport_height * (self.image_width as f32 / self.image_height as f32);

            self.w = unit_vector(subtract(self.lookfrom, self.lookat));
            self.u = unit_vector(cross(self.vup, self.w));
            self.v = cross(self.w, self.u);

            let viewport_u = scale(self.u, viewport_width);
            let viewport_v = scale(self.v, -viewport_height);

            self.pixel_delta_u = scale(viewport_u, 1.0 / self.image_width as f32);
            self.pixel_delta_v = scale(viewport_v, 1.0 / self.image_height as f32);

            let viewport_upper_left = subtract(
                subtract(
                    subtract(self.center, scale(self.w, focal_length)),
                    scale(viewport_u, 0.5),
                ),
                scale(viewport_v, 0.5),
            );

            self.pixel00_loc = add(
                viewport_upper_left,
                scale(add(self.pixel_delta_u, self.pixel_delta_v), 0.5),
            );
        }
    }

    unsafe impl DeviceRepr for Camera {}

    // impl Cameraf {
    //     pub fn new(aspect_ratio: FloatSize, image_width: u32) -> Self {
    //         let image_height = {
    //             let height = (image_width as FloatSize / aspect_ratio) as u32;
    //             if height == 0 {
    //                 1
    //             } else {
    //                 height
    //             }
    //         };
    //         image_height.max()
    //         let f_width = image_width as FloatSize;
    //         let f_height = image_height as FloatSize;
    //         let origin = [0.0f32; 3];
    //         let center = origin;
    //
    //         let vfov: FloatSize = 90.0;
    //         let theta: FloatSize = vfov.to_radians();
    //         let look_at = [0.0f32, 0.0, -1.0];
    //
    //         let diff_look = std::array::from_fn(|i| {
    //             (origin[i] - look_at[i])
    //         });
    //
    //         let focal_length = diff_look.map(|x| x.powi(2)).sum().sqrt();
    //
    //         let h = (theta / 2.0).tan();
    //         let vp_height = 2.0 * h * focal_length;
    //         let vp_width = vp_height * aspect_ratio;
    //
    //         let v_up = [0.0, 1.0, 0.0];
    //         let w = unit_vector(diff_look);
    //         let u = unit_vector(cross(v_up, w));
    //         let v = cross(v_up, w);
    //
    //         // calculate delta between pixel
    //
    //         // calculate the location of the top left pixel
    //         let top_left_pixel = [
    //             center[0] - vp_width / 2.0,
    //             center[1] + vp_height / 2.0,
    //             center[2] - focal_length,
    //         ];
    //
    //         let viewport = [
    //             viewport_width * u;
    //         let viewport = viewport_height * v;
    //         let pixel_delta = [viewport_u / f_width, -viewport_v / f_height, 0.0];
    //
    //         // let pixel00_loc: [FloatSize; 3] = std::array::from_fn(|i| {
    //         //     top_left_pixel[i] + pixel_delta[i] / 2.0
    //         // });
    //         let viewport_upper_left = std::array::from_fn(|i| {
    //             center[i] - focal_length * w[i] - viewport_u / 2 - viewport_v / 2
    //         });
    //         let pixel00_loc = top_left_pixel;
    //
    //         Self {
    //             aspect_ratio,
    //             image_width,
    //             image_height,
    //             vfov,
    //             center,
    //             pixel00_loc,
    //             pixel_delta,
    //         }
    //     }
    // }
}
