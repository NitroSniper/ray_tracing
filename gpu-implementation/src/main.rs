#![deny(clippy::all)]

mod ray_tracing;
mod gui;

use std::time::Instant;
use crate::ray_tracing::{CudaWorld};
use crate::gui::Framework;
use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use crate::ray_tracing::cuda_types::Camera;

const WIDTH: u32 = 1024;
const ASPECT_RATIO: f32 = 16.0 / 9.0;
fn main() -> Result<(), Error> {
    env_logger::init();
    let camera = Camera::new(ASPECT_RATIO, WIDTH, 8);
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, camera.image_height as f64);
        WindowBuilder::new()
            .with_title("GPU Implementation")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let (mut pixels, mut framework) = {
        let window_size = window.inner_size();
        let scale_factor = window.scale_factor() as f32;
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        let pixels = Pixels::new(camera.image_width, camera.image_height, surface_texture)?;
        let framework = Framework::new(
            &event_loop,
            window_size.width,
            window_size.height,
            scale_factor,
            &pixels,
        );
        (pixels, framework)
    };

    let mut cuda_world = CudaWorld::new(pixels.frame().len(), framework.gui.clone());
    let res = event_loop.run(|event, elwt| {

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(KeyCode::Escape) || input.close_requested() {
                elwt.exit();
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                }
                framework.resize(size.width, size.height);
            }

            // Update internal state and request a redraw
            window.request_redraw();
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let elapsed = Instant::now();
                framework.prepare(&window);
                cuda_world.render(pixels.frame_mut(), &camera);
                dbg!(elapsed.elapsed());

                let render_result = pixels.render_with(|encoder, render_target, context| {
                    context.scaling_renderer.render(encoder, render_target);
                    framework.render(encoder, render_target, context);
                    Ok(())
                });
                if let Err(err) = render_result {
                    log_error("pixels.render", err);
                    elwt.exit();
                    return;
                }
            },
            Event::WindowEvent {event, ..} => {
                framework.handle_event(&window, &event);
            }
            _ => ()
        }

    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}