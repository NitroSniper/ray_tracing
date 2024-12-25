#![allow(warnings)]
use std::{ops::Range, time::Instant};

use cgmath::Vector3;
use log::info;
use pixels::{Pixels, SurfaceTexture};
use ray_tracing::{Camera, Ray, Sphere, VectorRayExt};
use winit::{
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

static ASPECT_RATIO: f64 = 16.0 / 9.0;
static IMAGE_WIDTH: u32 = 400;
static SAMPLES: u32 = 100;
static MAX_DEPTH: u32 = 20;

fn main() {
    env_logger::init();
    // calculate image height

    // World
    let sphere_1 = Sphere::new(Vector3::new(0.0, 0.0, -1.0), 0.5);
    let sphere_2 = Sphere::new(Vector3::new(0.0, -100.5, -1.0), 100.0);
    let world = vec![sphere_1, sphere_2];

    // image
    let event_loop = EventLoop::new().unwrap();
    let _input = WinitInputHelper::new();
    event_loop.set_control_flow(ControlFlow::Poll);
    let camera = Camera::new(ASPECT_RATIO, IMAGE_WIDTH, SAMPLES);

    let window = {
        let size = LogicalSize::new(camera.image_width as u32, camera.image_height as u32);
        WindowBuilder::new()
            .with_title("Hell")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };
    let mut pixels = {
        let window_size = window.inner_size();
        dbg!(window_size);
        let surf = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(camera.image_width as u32, camera.image_height as u32, surf).unwrap()
    };

    let _ = event_loop.run(|event, elwt| match event {
        winit::event::Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => elwt.exit(),
        winit::event::Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            // Calculation
            let now = Instant::now();
            camera.render(pixels.frame_mut(), &world, MAX_DEPTH);
            info!("Calculation took {:?}", now.elapsed());
            pixels.render().unwrap();
            window.request_redraw();
        }
        _ => (),
    });
}
