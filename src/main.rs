use indicatif::ProgressBar;
use cgmath::{Array, InnerSpace, Vector3, Zero};
use ray_tracing::Ray;

fn main() {
    // image
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 256;
    
    // calculate image height
    let image_height = (image_width as f64 / aspect_ratio) as u64;
    let image_height = image_height.is_zero().then(|| 1).unwrap_or(image_height);

    // Camera
    let vp_height = 2.0;
    let vp_width =vp_height*(image_width as f64 / image_height as f64);
    let focal_length = 1.0;
    let camera_center = Vector3::new(0.0, 0.0, 0.0);

    // Calculate the vectors across the horizontal and down the vertical view edges.
    let vp_u = Vector3::new(vp_width, 0.0, 0.0);
    let vp_v = Vector3::new(0.0, -vp_height, 0.0);

    // calculate delta between pixel
    let pixel_delta_u = vp_u / image_width as f64;
    let pixel_delta_v = vp_v / image_height as f64;
    
    // calculate the location of the top left pixel
    let top_left_pixel = camera_center - Vector3::unit_z()*focal_length - vp_u/2.0 - vp_v/2.0;
    let pixel00_loc = top_left_pixel + 0.5 * (pixel_delta_u + pixel_delta_v);


    println!("P3\n{} {}\n255", image_width, image_height);
    let pb = ProgressBar::new(image_height * image_width);

    for j in 0..image_height {
        for i in 0..image_width {
            let pixel_center = pixel00_loc + pixel_delta_u*i as f64 + pixel_delta_v*j as f64;
            let ray = Ray::new(
                pixel_center,
                pixel_center - camera_center
            );
            write_color(ray_color(ray));
            pb.inc(1);
        }
    }
    pb.finish();
}

type Color = Vector3<f64>;
fn write_color(pixel_color: Color) {
    let pixel_max = 256.0 - 0.001;
    let i = pixel_max * pixel_color;
    println!("{} {} {}", i.x, i.y, i.z);
}

fn ray_color(ray: Ray<f64>) -> Color {
    let center = Vector3::new(0.0, 0.0, -1.0);
    if let Some(t) = hit_sphere(&center, 0.5, &ray) {
        let normal = (ray.at(t) - center).normalize();
        return (normal + Vector3::new(1.0, 1.0, 1.0)) * 0.5;
    }
    let unit_dir = ray.dir.normalize();
    let a = 0.5*(unit_dir.y + 1.0);
    (1.0-a)*Vector3::new(1.0, 1.0, 1.0)+ a*Vector3::new(0.5, 0.7, 1.0)
}

fn hit_sphere(center: &Vector3<f64>, rad: f64, ray: &Ray<f64>) -> Option<f64> {
    // https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection
    let oc = center - ray.orig;
    let a = ray.dir.dot(ray.dir);
    let h = ray.dir.dot(oc);
    let c = oc.dot(oc) - rad * rad;
    let discriminant = h*h - a*c;
    (discriminant >= 0.0).then(|| (h - discriminant.sqrt()) / a)
}