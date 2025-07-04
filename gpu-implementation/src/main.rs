#![deny(clippy::all)]

mod gui;
mod ray_tracing;

use crate::gui::Framework;
use crate::ray_tracing::camera::Camera;
use crate::ray_tracing::CudaWorld;
use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use std::rc::Rc;
use std::sync::RwLock;
use std::time::Instant;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: u32 = 1280;
const ASPECT_RATIO: f32 = 16.0 / 9.0;
fn main() -> Result<(), Error> {
    let camera: Rc<RwLock<Camera>> = RwLock::new(Camera::new(ASPECT_RATIO, WIDTH)).into();
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, camera.read().unwrap().image_height as f64);
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
        let pixels = Pixels::new(
            camera.read().unwrap().image_width,
            camera.read().unwrap().image_height,
            surface_texture,
        )?;
        let framework = Framework::new(
            &event_loop,
            window_size.width,
            window_size.height,
            scale_factor,
            &pixels,
            camera.clone(),
        );
        (pixels, framework)
    };
    let debug = framework.gui.debug.clone();
    let mut cuda_world = CudaWorld::new(pixels.frame().len(), debug.clone());

    let mut start = Instant::now();
    let mut cuda_render_ms = start.elapsed();
    let mut total_render_ms = start.elapsed();

    let res = event_loop.run(|event, elwt| {
        // Handle input events
        let _ = cuda_world.compile_ptx();
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
                framework.prepare(&window);
                start = Instant::now();
                cuda_world.render(pixels.frame_mut(), camera.clone());
                cuda_render_ms = start.elapsed();
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
                total_render_ms = start.elapsed();

                if !input.key_held(KeyCode::AltLeft) {
                    debug.write().expect("RwLock write fail").total_render_ms = total_render_ms;
                    debug.write().expect("RwLock write fail").cuda_render_ms = cuda_render_ms;
                }
            }
            Event::WindowEvent { event, .. } => {
                framework.handle_event(&window, &event);
            }
            _ => (),
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
