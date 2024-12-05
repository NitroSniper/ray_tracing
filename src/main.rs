use std::{ops::Range, time::Instant};

use cgmath::{num_traits::Float, Array, ElementWise, InnerSpace, Vector3};
use log::info;
use pixels::{Pixels, SurfaceTexture};
use rand::Rng;
use ray_tracing::{Ray, VectorRayExt};
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
static MAX_DEPTH: u32 = 10;

fn main() {
    env_logger::init();
    // calculate image height

    // World
    let sphere_1 = Sphere {
        center: Vector3::new(0.0, 0.0, -1.0),
        radius: 0.5,
    };
    let sphere_2 = Sphere {
        center: Vector3::new(0.0, -100.5, -1.0),
        radius: 100.0,
    };
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

type Color = Vector3<f64>;

struct HitRecord {
    point: Vector3<f64>,
    normal: Vector3<f64>,
    t: f64,
    front_face: bool,
}

impl Default for HitRecord {
    fn default() -> Self {
        Self {
            point: Vector3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 0.0),
            t: 0.0,
            front_face: false,
        }
    }
}

trait Hitter {
    fn hit(&self, ray: &Ray<f64>, t: &Range<f64>) -> Option<HitRecord>;
}

struct Sphere {
    center: Vector3<f64>,
    radius: f64,
}

impl Hitter for Sphere {
    fn hit(&self, ray: &Ray<f64>, t: &Range<f64>) -> Option<HitRecord> {
        let oc = self.center - ray.orig;
        let a = ray.dir.dot(ray.dir);
        let h = ray.dir.dot(oc);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();
        let root = {
            let root_a = (h - sqrtd) / a;
            let root_b = (h + sqrtd) / a;
            if t.contains(&root_a) {
                Some(root_a)
            } else if t.contains(&root_b) {
                Some(root_b)
            } else {
                None
            }
        }?;
        let mut record = HitRecord::default();
        record.t = root;
        record.point = ray.at(root);
        record.normal = (record.point - self.center) / self.radius;
        record.normal.invert_if_dot(&ray.dir, false);
        Some(record)
    }
}

impl<T: Hitter> Hitter for [T] {
    fn hit(&self, ray: &Ray<f64>, t: &Range<f64>) -> Option<HitRecord> {
        let mut record = None;
        let mut closest = t.end;
        for object in self {
            // Don't consider any object further
            let possible_record = object.hit(ray, &(t.start..closest));
            if let Some(obj_record) = possible_record {
                closest = obj_record.t;
                record = Some(obj_record);
            }
        }
        record
    }
}

struct Camera {
    aspect_ratio: f64,
    image_width: u32,
    image_height: u32,
    center: Vector3<f64>,
    pixel00_loc: Vector3<f64>,
    pixel_delta: Vector3<f64>,
    sample_per_pixel: u32,
}

impl Camera {
    fn new(aspect_ratio: f64, image_width: u32, sample_per_pixel: u32) -> Self {
        let image_height = {
            let height = (image_width as f64 / aspect_ratio) as u32;
            if height == 0 {
                1
            } else {
                height
            }
        };

        let f_width = image_width as f64;
        let f_height = image_height as f64;

        let vp_height = 2.0;
        let vp_width = vp_height * aspect_ratio;
        let focal_length = 1.0;
        let center = Vector3::from_value(0.0);

        // calculate delta between pixel
        let vp = Vector3::new(vp_width, -vp_height, 0.0);
        let pixel_delta = vp.div_element_wise(Vector3::new(f_width, f_height, 1.0));

        // calculate the location of the top left pixel
        let top_left_pixel = center - Vector3::unit_z() * focal_length - vp / 2.0;
        let pixel00_loc = top_left_pixel + pixel_delta / 2.0;

        Self {
            aspect_ratio,
            image_width,
            image_height,
            center,
            pixel00_loc,
            pixel_delta,
            sample_per_pixel,
        }
    }
    fn ray_color<T: Hitter>(depth: u32, ray: Ray<f64>, hittables: &[T]) -> Color {
        if depth <= 0 {
            return Vector3::from_value(0.0);
        }
        if let Some(record) = hittables.hit(&ray, &(0.0..f64::infinity())) {
            return 0.5
                * Camera::ray_color(
                    depth - 1,
                    Ray::new(
                        record.point,
                        *Vector3::random_unit_vector().invert_if_dot(&record.normal, true),
                    ),
                    hittables,
                );
        }
        let unit_dir = ray.dir.normalize();
        let a = 0.5 * (unit_dir.y + 1.0);
        (1.0 - a) * Vector3::from_value(1.0) + a * Vector3::new(0.5, 0.7, 1.0)
    }

    pub fn get_ray(&self, x: usize, y: usize, variance: Range<f64>) -> Ray<f64> {
        let mut rng = rand::thread_rng();
        let offset = Vector3::new(
            rng.gen_range(variance.start..variance.end),
            rng.gen_range(variance.start..variance.end),
            0.0,
        );
        let pixel = Vector3::new(x as f64, y as f64, 0.0);
        let pixel_center = (offset + pixel).mul_element_wise(self.pixel_delta) + self.pixel00_loc;
        let ray = Ray::new(self.center, pixel_center - self.center);
        ray
    }

    fn render<T: Hitter>(&self, frame: &mut [u8], world: &[T], max_depth: u32) {
        for (i, pixels) in frame.chunks_exact_mut(4).enumerate() {
            let x = i % self.image_width as usize;
            let y = i / self.image_width as usize;
            let color = (0..self.sample_per_pixel).fold(Vector3::from_value(0.0), |acc, _| {
                let ray = self.get_ray(x, y, -0.5..0.5);
                acc + Camera::ray_color(max_depth, ray, world)
            }) / self.sample_per_pixel as f64;

            // color is in 0..1 need to map to 0-255
            let u8_color: [u8; 4] = color.map(|x| (x * 255.0).round() as u8).extend(255).into();

            pixels.copy_from_slice(&u8_color);
        }
    }
}
