use cudarc::driver::{LaunchConfig, PushKernelArg};

const PTX_SRC: &str = include_str!("kernel.cu");
const SIZE: usize = 4;

fn main() {
    let ctx = cudarc::driver::CudaContext::new(0).expect("Failed to create cuda context");
    let stream = ctx.default_stream();

    // directly copied onto device
    let a: [f32; SIZE] = [1.0, 2.0, 3.0, 4.0];
    let b: [f32; SIZE] = [5.0, 6.0, 7.0, 8.0];
    let d_a = stream.memcpy_stod(&a).expect("Failed to create stream");
    let d_b = stream.memcpy_stod(&b).expect("Failed to create stream");
    // allocated directly on device
    let mut d_c = stream.alloc_zeros::<f32>(SIZE).expect("Failed to allocate stream");


    let ptx = cudarc::nvrtc::compile_ptx(PTX_SRC).expect("Failed to compile PTX");

    // load ptx

    let module = ctx.load_module(ptx).expect("Failed to load PTX");
    let sin_kernel = module.load_function("addKernel").expect("Failed to load kernel");

    let mut binding = stream.launch_builder(&sin_kernel);
    let builder = binding
        .arg(&mut d_c)
        .arg(&d_a)
        .arg(&d_b);

    unsafe {
        builder.launch(LaunchConfig::for_num_elems(SIZE as u32))
    }.expect("Failed to launch kernel");
    let c = stream.memcpy_dtov(&d_c).expect("Failed to return c stream");
    println!("{:?} + {:?} = {:?}", a, b, c);
}
