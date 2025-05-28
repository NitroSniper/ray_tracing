use std::any::Any;
use std::rc::Rc;
use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, DriverError, LaunchConfig, Profiler, PushKernelArg};
use std::sync::{Arc, RwLock};
use cudarc::curand::CudaRng;
use cudarc::nvrtc::CompileError;
use log::error;

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
    #[repr(C)]
    pub struct Camera {
        pub aspect_ratio: FloatSize,
        pub image_width: u32,
        pub image_height: u32,
        pub center: [FloatSize; 3],
        pub pixel00_loc: [FloatSize; 3],
        pub pixel_delta: [FloatSize; 3],
    }

    #[repr(C)]
    #[derive(PartialEq)]
    pub struct DeviceGUI {
        pub sample2_per_pixel: u32,
        pub block_dim: u32,
        pub max_depth: u32,
        pub show_random: bool,
        pub random_norm: bool
    }

    impl Default for DeviceGUI {
        fn default() -> Self {
            Self { show_random: false, random_norm: false, sample2_per_pixel: 4, block_dim: 1024, max_depth: 4 }
        }
    }
    unsafe impl DeviceRepr for Camera {}
    unsafe impl DeviceRepr for DeviceGUI {}
}

use cuda_types::Camera;
use cuda_types::FloatSize;
use crate::gui::{DebugGui, Gui};

pub struct CudaWorld {
    ctx: Arc<CudaContext>,
    render: Option<CudaFunction>,
    d_frame: CudaSlice<u8>,
    rng_block: CudaSlice<u32>,
    rng: CudaRng,
    gui: Rc<RwLock<DebugGui>>,
}

// const PTX_SRC: &str = concat!(include_str!("cuda/floatN_helper.cu"), include_str!("cuda/lib.cu"), include_str!("cuda/kernel.cu"));
const PTX_SRC: &str = concat!(include_str!("cuda/floatN_helper.cu"), include_str!("cuda/library.cu"), include_str!("cuda/ray.cu"));

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
        let ctx = cudarc::driver::CudaContext::new(0).expect("Failed to create CudaContext");
        let stream = ctx.default_stream();
        let rng = CudaRng::new(0, stream.clone()).expect("Failed to create CudaRng");
        let mut rng_block = stream.alloc_zeros::<u32>(frame_size).expect("Failed to allocate frame buffer on device");
        rng.fill_with_uniform(&mut rng_block).expect("Failed to fill rng block");
        let d_frame = stream.alloc_zeros::<u8>(frame_size).expect("Failed to allocate frame buffer on device");

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
                Err(CompileError::CompileError { nvrtc: _, options: _, log }) => {
                    error!("{}", log.as_c_str().to_str().unwrap());
                    Err(PtxError::CompileError)
                },
                Err(err) => Err(PtxError::CompileErrorOther(err)),
                Ok(ptx) => {
                    let module = self.ctx.load_module(ptx).expect("Failed to load PTX");
                    module.load_function("render").map_err(PtxError::LoadFunction)
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

    pub fn render(&mut self, frame: &mut [u8], camera: &Camera) {
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
            self.rng.fill_with_uniform(&mut self.rng_block).expect("Failed to fill rng block");
            self.gui.write().expect("Gui can't be written").random = false;
        };

        let launch = {
            let gui = self.gui.read().expect("Gui can't be read");
            let builder = binding.arg(&self.rng_block).arg(&mut self.d_frame).arg(camera).arg(&gui.device_gui);
            let dim = gui.device_gui.block_dim;
            let launch_cfg = LaunchConfig {block_dim: (dim, 1, 1), grid_dim: ((camera.image_width * camera.image_height).div_ceil(gui.device_gui.block_dim), 1, 1), shared_mem_bytes: 0};
            unsafe { builder.launch(launch_cfg) }
        };
        if let Err(err) = launch {
            self.gui.write().unwrap().render_msg = err.error_name().unwrap().to_str().unwrap().into();
        } else {
            stream.memcpy_dtoh(&self.d_frame, frame).expect("Failed to copy device frames");
        }
    }
}

impl Camera {
    pub fn new(aspect_ratio: FloatSize, image_width: u32) -> Self {
        let image_height = {
            let height = (image_width as FloatSize / aspect_ratio) as u32;
            if height == 0 {
                1
            } else {
                height
            }
        };
        let f_width = image_width as FloatSize;
        let f_height = image_height as FloatSize;

        let vp_height = 2.0;
        let vp_width = vp_height * aspect_ratio;
        let focal_length = 1.0;
        let center = [0.0; 3];

        // calculate delta between pixel
        let pixel_delta = [vp_width / f_width, -vp_height / f_height, 0.0];

        // calculate the location of the top left pixel
        let top_left_pixel = [
            center[0] - vp_width / 2.0,
            center[1] + vp_height / 2.0,
            center[2] - focal_length,
        ];

        // let pixel00_loc: [FloatSize; 3] = std::array::from_fn(|i| {
        //     top_left_pixel[i] + pixel_delta[i] / 2.0
        // });
        let pixel00_loc = top_left_pixel;

        Self {
            aspect_ratio,
            image_width,
            image_height,
            center,
            pixel00_loc,
            pixel_delta,
        }
    }
}