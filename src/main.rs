use std::ops::Range;

use cgmath::{num_traits::Float, Array, InnerSpace, Vector3};
use pixels::{Pixels, SurfaceTexture};
use ray_tracing::Ray;
use winit::{
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

static ASPECT_RATIO: f64 = 16.0 / 9.0;
static IMAGE_WIDTH: u32 = 256;

fn main() {
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
    let camera = Camera::new(ASPECT_RATIO, IMAGE_WIDTH);

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
            camera.render(pixels.frame_mut(), &world);
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

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray<f64>, out_norm: &Vector3<f64>) {
        self.front_face = ray.dir.dot(*out_norm) < 0.0;
        self.normal = if self.front_face {
            *out_norm
        } else {
            -1.0 * out_norm
        };
    }
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
        let out_normal = (record.point - self.center) / self.radius;
        record.set_face_normal(ray, &out_normal);
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
    pixel_delta_u: Vector3<f64>,
    pixel_delta_v: Vector3<f64>,
}

impl Camera {
    fn new(aspect_ratio: f64, image_width: u32) -> Self {
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
        let vp_width = vp_height * (f_width / f_height);
        let focal_length = 1.0;
        let center = Vector3::from_value(0.0);

        // Calculate the vectors across the horizontal and down the vertical view edges.
        let vp_u = Vector3::unit_x() * vp_width;
        let vp_v = Vector3::unit_y() * -vp_height;

        // calculate delta between pixel
        let pixel_delta_u = vp_u / f_width;
        let pixel_delta_v = vp_v / f_height;

        // calculate the location of the top left pixel
        let top_left_pixel = center - Vector3::unit_z() * focal_length - vp_u / 2.0 - vp_v / 2.0;
        let pixel00_loc = top_left_pixel + 0.5 * (pixel_delta_u + pixel_delta_v);

        Self {
            aspect_ratio,
            image_width,
            image_height,
            center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }
    fn ray_color<T: Hitter>(ray: Ray<f64>, hittables: &[T]) -> Color {
        if let Some(record) = hittables.hit(&ray, &(0.0..f64::infinity())) {
            return 0.5 * (record.normal + Vector3::from_value(1.0));
        }
        let unit_dir = ray.dir.normalize();
        let a = 0.5 * (unit_dir.y + 1.0);
        (1.0 - a) * Vector3::from_value(1.0) + a * Vector3::new(0.5, 0.7, 1.0)
    }

    fn render<T: Hitter>(&self, frame: &mut [u8], world: &[T]) {
        for (i, pixels) in frame.chunks_exact_mut(4).enumerate() {
            let x = i % IMAGE_WIDTH as usize;
            let y = i / IMAGE_WIDTH as usize;
            let pixel_center =
                self.pixel00_loc + self.pixel_delta_u * x as f64 + self.pixel_delta_v * y as f64;
            let ray = Ray::new(self.center, pixel_center - self.center);
            let color = Camera::ray_color(ray, world);
            let u8_color = color.map(|x| (x * 255.0).round() as u8);
            pixels.copy_from_slice(&[u8_color.x, u8_color.y, u8_color.z, 0xff]);
        }
    }
}
