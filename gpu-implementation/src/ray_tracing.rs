use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, Profiler, PushKernelArg};
use std::sync::Arc;
use cudarc::curand::CudaRng;

pub mod cuda_types {
    use cudarc::driver::DeviceRepr;

    pub type FloatSize = f32;

    struct Pixel {
        r: FloatSize,
        g: FloatSize,
        b: FloatSize,
        a: FloatSize,
    }
    pub struct Camera {
        pub aspect_ratio: FloatSize,
        pub image_width: u32,
        pub image_height: u32,
        pub samples_per_pixel: u32,
        pub center: [FloatSize; 3],
        pub pixel00_loc: [FloatSize; 3],
        pub pixel_delta: [FloatSize; 3],
    }
    unsafe impl DeviceRepr for Camera {}
}

use cuda_types::Camera;
use cuda_types::FloatSize;


pub struct CudaWorld {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    render: CudaFunction,
    d_frame: CudaSlice<FloatSize>,
    rng_block: CudaSlice<u32>,
    rng: CudaRng,
}

const PTX_SRC: &str = concat!(include_str!("cuda/floatN_helper.cu"), include_str!("cuda/lib.cu"), include_str!("cuda/kernel.cu"));

impl CudaWorld {
    pub fn new(frame_size: usize) -> Self {
        dbg!(PTX_SRC);
        let ctx = cudarc::driver::CudaContext::new(0).expect("Failed to create CudaContext");
        let stream = ctx.default_stream();
        let rng = CudaRng::new(0, stream.clone()).expect("Failed to create CudaRng");
        let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC).unwrap_or_else(|err| {
            use cudarc::nvrtc::CompileError;
            match err {
                CompileError::CompileError { nvrtc: _, options: _, log } => unsafe{ panic!("{}", log.as_c_str().to_str().unwrap()) },
                _ => panic!("Failed to compile PTX: {:#?}", err)
            };
        });

        let module = ctx.load_module(ptx).expect("Failed to load PTX");
        let render = module.load_function("render").expect("Failed to load render function");
        let rng_block = stream.alloc_zeros::<u32>(frame_size).expect("Failed to allocate frame buffer on device");
        let d_frame = stream.alloc_zeros::<FloatSize>(frame_size).expect("Failed to allocate frame buffer on device");

        Self {
            ctx,
            module,
            render,
            d_frame,
            rng_block,
            rng
        }
    }

    pub fn render(&mut self, frame: &mut [u8], camera: &Camera) {
        let stream = self.ctx.default_stream();
        let mut binding = stream.launch_builder(&self.render);
        self.rng.fill_with_uniform(&mut self.rng_block).expect("Failed to fill rng block");
        let builder = binding.arg(&self.rng_block).arg(&mut self.d_frame).arg(camera);
        unsafe {
            builder.launch(LaunchConfig::for_num_elems(camera.image_width * camera.image_height))
        }.expect("Failed to launch kernel");

        let float_frames = stream.memcpy_dtov(&self.d_frame).expect("Failed to copy device frames");
        frame.iter_mut().zip(float_frames.iter()).for_each(|(dst, src)| {
            *dst = src.clamp(0.0, 255.0) as u8;
        });
    }
}

impl Camera {
    pub fn new(aspect_ratio: FloatSize, image_width: u32, samples_per_pixel: u32) -> Self {
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
            samples_per_pixel,
        }
    }
}