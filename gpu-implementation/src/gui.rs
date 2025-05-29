use crate::ray_tracing::camera::Camera;
use crate::ray_tracing::cuda_types::DeviceGUI;
use egui::{ClippedPrimitive, Context, TexturesDelta, Ui, ViewportId, Widget};
use egui_wgpu::{Renderer, ScreenDescriptor};
use pixels::{wgpu, PixelsContext};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::RwLock;
use std::time::Duration;
use winit::event_loop::EventLoopWindowTarget;
use winit::window::Window;

/// Manages all state required for rendering egui over `Pixels`.
pub(crate) struct Framework {
    // State for egui.
    egui_ctx: Context,
    egui_state: egui_winit::State,
    screen_descriptor: ScreenDescriptor,
    renderer: Renderer,
    paint_jobs: Vec<ClippedPrimitive>,
    textures: TexturesDelta,

    // State for the GUI
    pub gui: Gui,
}

impl Framework {
    /// Create egui.
    pub(crate) fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        width: u32,
        height: u32,
        scale_factor: f32,
        pixels: &pixels::Pixels,
        camera: Rc<RwLock<Camera>>,
    ) -> Self {
        let max_texture_size = pixels.device().limits().max_texture_dimension_2d as usize;

        let egui_ctx = Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            Some(scale_factor),
            Some(max_texture_size),
        );
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: scale_factor,
        };
        let renderer = Renderer::new(pixels.device(), pixels.render_texture_format(), None, 1);
        let textures = TexturesDelta::default();
        let gui = Gui::new(camera);

        Self {
            egui_ctx,
            egui_state,
            screen_descriptor,
            renderer,
            paint_jobs: Vec::new(),
            textures,
            gui,
        }
    }

    /// Handle input events from the window manager.
    pub(crate) fn handle_event(&mut self, window: &Window, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_window_event(window, event);
    }

    /// Resize egui.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.screen_descriptor.size_in_pixels = [width, height];
        }
    }

    /// Update scaling factor.
    pub(crate) fn scale_factor(&mut self, scale_factor: f64) {
        self.screen_descriptor.pixels_per_point = scale_factor as f32;
    }

    /// Prepare egui.
    pub(crate) fn prepare(&mut self, window: &Window) {
        // Run the egui frame and create all paint jobs to prepare for rendering.
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run(raw_input, |egui_ctx| {
            // Draw the demo application.
            self.gui.ui(egui_ctx);
        });

        self.textures.append(output.textures_delta);
        self.egui_state
            .handle_platform_output(window, output.platform_output);
        self.paint_jobs = self
            .egui_ctx
            .tessellate(output.shapes, self.screen_descriptor.pixels_per_point);
    }

    /// Render egui.
    pub(crate) fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        render_target: &wgpu::TextureView,
        context: &PixelsContext,
    ) {
        // Upload all resources to the GPU.
        for (id, image_delta) in &self.textures.set {
            self.renderer
                .update_texture(&context.device, &context.queue, *id, image_delta);
        }
        self.renderer.update_buffers(
            &context.device,
            &context.queue,
            encoder,
            &self.paint_jobs,
            &self.screen_descriptor,
        );

        // Render egui with WGPU
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer
                .render(&mut rpass, &self.paint_jobs, &self.screen_descriptor);
        }

        // Cleanup
        let textures = std::mem::take(&mut self.textures);
        for id in &textures.free {
            self.renderer.free_texture(id);
        }
    }
}

/// Example application state. A real application will need a lot more state than this.
pub struct Gui {
    /// Only show the egui window when true.
    window_open: bool,
    pub debug: Rc<RwLock<DebugGui>>,
    camera: Rc<RwLock<Camera>>,
}

impl Gui {
    fn new(camera: Rc<RwLock<Camera>>) -> Self {
        Self {
            window_open: false,
            debug: Rc::new(RwLock::new(DebugGui::default())),
            camera,
        }
    }
}

impl Gui {
    /// Create the UI using egui.
    fn ui(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("menubar_container").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Debug").clicked() {
                        self.window_open = true;
                        ui.close_menu();
                    }
                })
            });
        });

        egui::Window::new("Debug Menu")
            .open(&mut self.window_open)
            .show(ctx, |ui| unsafe {
                self.debug
                    .write()
                    .expect("Fetch inner debug menu")
                    .ui(ui, self.camera.clone());
            });
    }
}

#[derive(PartialEq)]
pub struct DebugGui {
    pub render_msg: Rc<str>,
    pub random: bool,
    pub total_render_ms: Duration,
    pub cuda_render_ms: Duration,
    pub compile_ptx: bool,
    pub device_gui: DeviceGUI,
}
impl Default for DebugGui {
    fn default() -> Self {
        Self {
            render_msg: "".into(),
            random: false,
            total_render_ms: Duration::new(0, 0),
            cuda_render_ms: Duration::new(0, 0),
            compile_ptx: true,
            device_gui: Default::default(),
        }
    }
}

impl DebugGui {
    unsafe fn ui(&mut self, ui: &mut Ui, camera: Rc<RwLock<Camera>>) {
        ui.label("This is the debug menu for the ray tracer.");
        ui.separator();
        ui.heading("Statistic");
        ui.label("Hold `LeftAlt` down to pause statistic");
        ui.add_space(8.0);
        ui.label(format!("Last Error: {:?}", self.render_msg));
        ui.label(format!("Total Render Duration: {:?}", self.total_render_ms));
        ui.label(format!(
            "Total Cuda Render Duration: {:?}",
            self.cuda_render_ms
        ));
        ui.separator();
        ui.heading("Camera");
        let mut camera = camera.write().unwrap();
        ui.add(
            egui::Slider::new(&mut camera.vfov, 0.1..=100.0)
                .text("FOV")
                .logarithmic(true)
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookfrom[0], -100.0..=100.0)
                .text("Camera X")
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookfrom[1], -10.0..=10.0)
                .text("Camera Y")
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookfrom[2], -100.0..=100.0)
                .text("Camera Z")
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookat[0], -100.0..=100.0)
                .text("Destination X")
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookat[1], -10.0..=10.0)
                .text("Destination Y")
                .clamp_to_range(true),
        );
        ui.add(
            egui::Slider::new(&mut camera.lookat[2], -100.0..=100.0)
                .text("Destination Z")
                .clamp_to_range(true),
        );

        ui.separator();
        ui.heading("Options");
        ui.checkbox(&mut self.device_gui.show_random, "Show Random State?");
        ui.checkbox(&mut self.device_gui.random_norm, "Normalise Random State?");
        ui.add({
            let samples = self.device_gui.sample2_per_pixel.pow(2);
            egui::Slider::new(&mut self.device_gui.sample2_per_pixel, 1..=50)
                .text(format!("{:?} rays per pixel (sample^2 per pixel)", samples))
                .logarithmic(true)
                .clamp_to_range(true)
                .trailing_fill(true)
        });
        ui.add(
            egui::Slider::new(&mut self.device_gui.max_depth, 1..=50)
                .text("Max Light Bounces")
                .logarithmic(true)
                .clamp_to_range(true)
                .trailing_fill(true),
        );
        ui.add(
            egui::Slider::new(&mut self.device_gui.block_dim, 128..=1024)
                .text("Block Dim Amount")
                .clamp_to_range(true)
                .trailing_fill(true),
        );
        if ui.button("Generate New Random State").clicked() {
            self.random = true;
        }
        if ui.button("Compile PTX Again?").clicked() {
            self.compile_ptx = true;
        }
        ui.heading("Presets");
        ui.separator();
        if ui.button("Go To Reflection Test").clicked() {
            camera.lookat = [0.0, 0.0, -70.0];
            camera.lookfrom = [0.0, 0.0, -85.0];
        }
        if ui.button("Go To Starting Test").clicked() {
            camera.lookat = [0.0, 0.0, -1.0];
            camera.lookfrom = [-2.0, 2.0, 1.0];
        }

        ui.add_space(8.0);
        ui.vertical_centered(|ui| {
            egui::reset_button(ui, self);
        });
    }
}
