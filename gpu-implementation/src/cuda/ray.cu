__device__ float3 ray_color(Ray r, Sphere* world, unsigned int world_size) {
    float2 t = make_float2(0.0f, 1024.0f);

    // float3 color = make_float3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < world_size; i++) {
        Sphere& s = world[i];
        HitRecord record = s.hit(r, t);
        if (!record.is_none) {
            float3 n = normalize(sub(r.at(record.t), make_float3(0.0, 0.0, -1.0)));
            return mul(make_float3(n.x+1.0, n.y+1.0,n.z+1.0), 0.5);
        }
    }

    float3 unit_dir = normalize(r.dir);
    float a = (unit_dir.y + 1.0f) * 0.5f;
    return add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a));
}


__device__ Ray get_ray(
    const unsigned int i,
    const unsigned int j,
    const unsigned int sample,
    const unsigned int sample_len,
    const Camera& cam
) {
	float sample_i = (float)(sample % sample_len) / sample_len;
	float sample_j = (float)(sample / sample_len) / sample_len;

    float3 pixel_center = add(mul(make_float3((float)i+sample_i, (float)j+sample_j, 0.0f), cam.pixel_delta), cam.pixel00_loc);
    Ray r(cam.center, sub(pixel_center, cam.center));
    return r;
}


__device__ uchar4 to_pixel(float4 fpixel) {
    uchar4 c = make_uchar4(
        (unsigned char)(fmaxf(0.0f, fminf(fpixel.x * 255.0f, 255.0f))),
        (unsigned char)(fmaxf(0.0f, fminf(fpixel.y * 255.0f, 255.0f))),
        (unsigned char)(fmaxf(0.0f, fminf(fpixel.z * 255.0f, 255.0f))),
        (unsigned char)(fmaxf(0.0f, fminf(fpixel.w * 255.0f, 255.0f)))
    );
    return c;
}


extern "C" __global__ void render(uint64_t *rng_state, uchar4 *const frame, Camera cam_g, GuiState gui_g) {
    // Load shared data
    __shared__ Camera cam;
    __shared__ Sphere s;
    __shared__ GuiState gui;
    const unsigned int world_size = 2;
    __shared__ Sphere world[world_size];
    cam = static_cast<Camera&&>(cam_g);
    gui = static_cast<GuiState&&>(gui_g);
    world[0] = Sphere(make_float3(0.0f,0.0f,-1.2f), 0.5f);
    world[1] = Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f);
    __syncthreads();

    // if thread doesn't have a pixel
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
    if (idx >= cam.image_width*cam.image_height) return;

	pcg32_global = {rng_state[idx], 0};

    if (gui.show_random) {
	    float3 rgb = gui.random_norm ? random_norm_float3() : random_float3();
	    frame[idx] = to_pixel(make_float4_f3(rgb, 1.0f));
	    return;
    }

    // Ray Color
	int i = idx % cam.image_width;
	int j = idx / cam.image_width;
    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    unsigned int samples = gui.sample2_per_pixel*gui.sample2_per_pixel;
    for (int sample_idx = 0; sample_idx < samples; sample_idx++) {
        total = add(total, ray_color(get_ray(i, j, sample_idx, gui.sample2_per_pixel, cam), world, world_size));
        __syncthreads();
    }

    frame[idx] = to_pixel(make_float4_f3(div(total, samples), 1.0f));
}
