use indicatif::ProgressBar;
use cgmath::Vector3;

fn main() {
    let image_width = 256;
    let image_height = 256;
    println!("P3\n{} {}\n255", image_width, image_height);
    let pb = ProgressBar::new(image_height * image_width);

    for j in 0..image_height {
        for i in 0..image_width {
            let pixel_color = Vector3::new(
                i as f64 / (image_width-1) as f64,
                j as f64 / (image_width-1) as f64,
                0.25);

            write_color(pixel_color);
            pb.inc(1);
        }
    }
    pb.finish();
}

type Color = Vector3<f64>;
fn write_color(pixel_color: Color) {
    let pixel_max = 256.0 - 0.001;
    let ir = (pixel_max * pixel_color.x) as u8;
    let ig = (pixel_max * pixel_color.y) as u8;
    let ib = (pixel_max * pixel_color.z) as u8;
    println!("{} {} {}", ir, ig, ib);
    
}