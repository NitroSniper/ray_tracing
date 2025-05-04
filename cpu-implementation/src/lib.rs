use std::ops::Range;

use cgmath::{num_traits::Float, Array, BaseFloat, ElementWise, InnerSpace, Vector3};
use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub struct Ray<S> {
    pub orig: Vector3<S>,
    pub dir: Vector3<S>,
}

impl<S: BaseFloat> Ray<S> {
    pub fn new(orig: Vector3<S>, dir: Vector3<S>) -> Self {
        Self { orig, dir }
    }

    pub fn at(&self, lambda: S) -> Vector3<S> {
        self.orig + self.dir * lambda
    }
}

pub trait VectorRayExt<S> {
    fn random_unit_vector() -> Vector3<S>;
    fn invert_if_dot(&mut self, rhs: &Vector3<S>, is_negative: bool) -> &mut Self;
}

impl<S: BaseFloat> VectorRayExt<S> for Vector3<S> {
    fn random_unit_vector() -> Vector3<S> {
        let mut rng = rand::thread_rng();
        loop {
            let p = Vector3::<S>::new(
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
                S::from(rng.gen_range(-1.0..=1.0))
                    .expect("-1.0..1.0 should be viable as BaseFloat"),
            );
            // check if it is within circle; this filter removes any bias
            if p.dot(p) <= S::one() {
                return p.normalize();
            }
        }
    }

    fn invert_if_dot(&mut self, rhs: &Vector3<S>, is_negative: bool) -> &mut Vector3<S> {
        let val = self.dot(*rhs);
        let minus_one = -S::one();
        match (val.is_sign_negative(), is_negative) {
            (true, true) => *self *= minus_one,
            (false, false) => *self *= minus_one,
            _ => (),
        };
        self
    }
}

type Color = Vector3<f64>;

pub struct HitRecord {
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

pub trait Hitter {
    fn hit(&self, ray: &Ray<f64>, t: &Bound) -> Option<HitRecord>;
}

pub struct Sphere {
    center: Vector3<f64>,
    radius: f64,
}

impl Sphere {
    pub fn new(center: Vector3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Hitter for Sphere {
    fn hit(&self, ray: &Ray<f64>, t: &Bound) -> Option<HitRecord> {
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
            if t.contains(root_a) {
                Some(root_a)
            } else if t.contains(root_b) {
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
    fn hit(&self, ray: &Ray<f64>, t: &Bound) -> Option<HitRecord> {
        self.iter().fold(None, |record, object| {
            object
                .hit(ray, &t.clamp_end(record.as_ref().map(|r| r.t), t.end))
                .or(record)
        })
    }
}

pub struct Camera {
    pub aspect_ratio: f64,
    pub image_width: u32,
    pub image_height: u32,
    pub center: Vector3<f64>,
    pub pixel00_loc: Vector3<f64>,
    pub pixel_delta: Vector3<f64>,
    pub sample_per_pixel: u32,
}

impl Camera {
    pub fn new(aspect_ratio: f64, image_width: u32, sample_per_pixel: u32) -> Self {
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
    pub fn ray_color<T: Hitter>(
        depth: u32,
        ray: Ray<f64>,
        hittables: &[T],
        reflectance: f64,
    ) -> Color {
        if depth <= 0 {
            return Vector3::from_value(0.0);
        }
        if let Some(record) = hittables.hit(&ray, &Bound::new(0.001, f64::infinity())) {
            let dir = record.normal + Vector3::random_unit_vector();
            return reflectance
                * Camera::ray_color(
                    depth - 1,
                    Ray::new(record.point, dir),
                    hittables,
                    reflectance,
                );
        }
        let unit_dir = ray.dir.normalize();
        let a = reflectance * (unit_dir.y + 1.0);
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

    pub fn render<T: Hitter + Sync>(&self, frame: &mut [u8], world: &[T], max_depth: u32) {
        frame.par_chunks_mut(4).enumerate().for_each(|(i, pixels)| {
            let x = i % self.image_width as usize;
            let y = i / self.image_width as usize;
            let gamma_index = self.image_width / 5;
            let gamma_section = 0.1 + 0.2 * (x as u32 / gamma_index) as f64;

            let color = (0..self.sample_per_pixel).fold(Vector3::from_value(0.0), |acc, _| {
                let ray = self.get_ray(x, y, -0.5..0.5);
                acc + Camera::ray_color(max_depth, ray, world, gamma_section)
            }) / self.sample_per_pixel as f64;

            // color is in 0..1 need to map to 0-255
            // Square root is gamma correction
            let u8_color: [u8; 4] = color
                .map(|x| (x.sqrt() * 255.0).round() as u8)
                .extend(255)
                .into();

            pixels.copy_from_slice(&u8_color);
        })
    }
}

pub struct Bound {
    begin: f64,
    end: f64,
}

impl Bound {
    fn new(begin: f64, end: f64) -> Self {
        Self { begin, end }
    }

    fn contains(&self, t: f64) -> bool {
        self.begin < t && t < self.end
    }

    fn clamp_end(&self, t: Option<f64>, default: f64) -> Bound {
        Bound {
            end: t.unwrap_or(default),
            ..*self
        }
    }
}
