use std::iter::zip;
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

pub mod cuda_types {
    pub type FloatSize = f32;

    struct Pixel {
        r: FloatSize,
        g: FloatSize,
        b: FloatSize,
        a: FloatSize,
    }
}

use cuda_types::FloatSize;


pub struct Camera {
    pub aspect_ratio: FloatSize,
    pub image_width: u32,
    pub image_height: u32,
    pub center: [FloatSize; 3],
    pub pixel00_loc: [FloatSize; 3],
    pub pixel_delta: [FloatSize; 3],
    pub samples_per_pixel: u32,
}

pub struct CudaWorld {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    render: CudaFunction,
    d_frames: CudaSlice<FloatSize>,
}

const PTX_SRC: &str = include_str!("kernel.cu");

impl CudaWorld {
    pub fn new(frame_size: usize) -> Self {
        let ctx = cudarc::driver::CudaContext::new(0).expect("Failed to create CudaContext");
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC).expect("Failed to compile PTX");
        let module = ctx.load_module(ptx).expect("Failed to load PTX");
        let render = module.load_function("render").expect("Failed to load render function");

        let d_frames = stream.alloc_zeros::<FloatSize>(frame_size).expect("Failed to allocate frame buffer on device");

        Self {
            ctx,
            module,
            render,
            d_frames
        }
    }

    pub fn render(&mut self, frames: &mut [u8]) {
        let stream = self.ctx.default_stream();
        let mut binding = stream.launch_builder(&self.render);
        let builder = binding.arg(&mut self.d_frames);
        let len = frames.len();

        unsafe {
            builder.launch(LaunchConfig::for_num_elems(len as u32))
        }.expect("Failed to launch kernel");

        let float_frames = stream.memcpy_dtov(&self.d_frames).expect("Failed to copy device frames");
        frames.iter_mut().zip(float_frames.iter()).for_each(|(dst, src)| {
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
            center[0] - focal_length - vp_width / 2.0,
            center[1] - focal_length - vp_height / 2.0,
            center[2] - focal_length,
        ];

        let pixel00_loc: [FloatSize; 3] = std::array::from_fn(|i| {
            top_left_pixel[i] - pixel_delta[i] / 2.0
        });

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