use cgmath::{num_traits::{Float, FromPrimitive}, ElementWise, InnerSpace, Vector3, Zero};
use indicatif::ProgressBar;
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
static IMAGE_HEIGHT: u32 = {
    let height = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as u32;
    if height == 0 {
        1
    } else {
        height
    }
};
fn main() {
    // calculate image height

    // image
    let event_loop = EventLoop::new().unwrap();
    let input = WinitInputHelper::new();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = {
        let size = LogicalSize::new(IMAGE_WIDTH, IMAGE_HEIGHT);
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
        Pixels::new(IMAGE_WIDTH, IMAGE_HEIGHT, surf).unwrap()
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
            draw_image(pixels.frame_mut());
            pixels.render().unwrap();
            window.request_redraw();
        },
        _ => (),
    });
}

fn draw_image(frame: &mut [u8]) {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 256;

    // calculate image height
    let image_height = (image_width as f64 / aspect_ratio) as u64;
    let image_height = image_height.is_zero().then(|| 1).unwrap_or(image_height);

    // Camera
    let vp_height = 2.0;
    let vp_width = vp_height * (image_width as f64 / image_height as f64);
    let focal_length = 1.0;
    let camera_center = Vector3::new(0.0, 0.0, 0.0);

    // Calculate the vectors across the horizontal and down the vertical view edges.
    let vp_u = Vector3::new(vp_width, 0.0, 0.0);
    let vp_v = Vector3::new(0.0, -vp_height, 0.0);

    // calculate delta between pixel
    let pixel_delta_u = vp_u / image_width as f64;
    let pixel_delta_v = vp_v / image_height as f64;

    // calculate the location of the top left pixel
    let top_left_pixel = camera_center - Vector3::unit_z() * focal_length - vp_u / 2.0 - vp_v / 2.0;
    let pixel00_loc = top_left_pixel + 0.5 * (pixel_delta_u + pixel_delta_v);


    for (i, pixels) in frame.chunks_exact_mut(4).enumerate() {
        let x = i % IMAGE_WIDTH as usize;
        let y = i / IMAGE_WIDTH as usize;
        let pixel_center = pixel00_loc + pixel_delta_u * x as f64 + pixel_delta_v * y as f64;
        let ray = Ray::new(pixel_center, pixel_center - camera_center);
        let color = ray_color(ray);
        let u8_color = color.map(|x| (x * 255.0).round() as u8);
        pixels.copy_from_slice(
            &[u8_color.x, u8_color.y, u8_color.z, 0xff]
        );
    }
}

type Color = Vector3<f64>;
fn ray_color(ray: Ray<f64>) -> Color {
    let center = Vector3::new(0.0, 0.0, -1.0);
    if let Some(t) = hit_sphere(&center, 0.5, &ray) {
        let normal = (ray.at(t) - center).normalize();
        return (normal + Vector3::new(1.0, 1.0, 1.0)) * 0.5;
    }
    let unit_dir = ray.dir.normalize();
    let a = 0.5 * (unit_dir.y + 1.0);
    (1.0 - a) * Vector3::new(1.0, 1.0, 1.0) + a * Vector3::new(0.5, 0.7, 1.0)
}

fn hit_sphere(center: &Vector3<f64>, rad: f64, ray: &Ray<f64>) -> Option<f64> {
    // https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection
    let oc = center - ray.orig;
    let a = ray.dir.dot(ray.dir);
    let h = ray.dir.dot(oc);
    let c = oc.dot(oc) - rad * rad;
    let discriminant = h * h - a * c;
    (discriminant >= 0.0).then(|| (h - discriminant.sqrt()) / a)

}

struct HitRecord {
    point: Vector3<f64>,
    normal: Vector3<f64>,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray<f64>, out_norm: &Vector3<f64>) {
        self.front_face = ray.dir.dot(*out_norm) < 0.0;
        self.normal = if self.front_face {*out_norm} else {-1.0 * out_norm};
    }
}

impl Default for HitRecord {
    fn default() -> Self {
        Self { 
            point: Vector3::new(0.0,0.0,0.0),
            normal: Vector3::new(0.0,0.0,0.0),
            t: 0.0,
            front_face: false,
        }
    }
}


trait Hitter {
    fn hit(&self, ray: &Ray<f64>, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

struct Sphere {
    center: Vector3<f64>,
    radius: f64,
}

impl Hitter for Sphere {
    fn hit(&self, ray: &Ray<f64>, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = self.center - ray.orig;
        let a = ray.dir.dot(ray.dir);
        let h = ray.dir.dot(oc);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = h*h - a*c;
        if discriminant < 0.0 {
            return None
        }
        let sqrtd = discriminant.sqrt();
        let root = {
            let root_a = h-sqrtd / a;
            let root_b = h+sqrtd / a;
            if (t_min..t_max).contains(&root_a){
                Some(root_a)
            } else if (t_min..t_max).contains(&root_b) {
                Some(root_b)
            } else {
                None
            }
        }?;
        let mut record = HitRecord::default();
        record.t = root;
        record.point = ray.at(record.t);
        let out_normal = (record.point - self.center) / self.radius;
        record.set_face_normal(ray, &out_normal);
        Some(record) 
    }
}
