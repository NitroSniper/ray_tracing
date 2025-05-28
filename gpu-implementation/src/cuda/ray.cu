__device__ float3 ray_color(Ray r) {
     float3 unit_dir = normalize(r.dir);
     float a = (unit_dir.y + 1.0f) * 0.5f;
     float3 color = add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a));
     return color;
}


__device__ Ray get_ray(
    const unsigned int i,
    const unsigned int j,
    const unsigned int sample,
    const Camera& cam
) {
	float sample_i = (float)(sample % cam.samples_per_pixel) / cam.samples_per_pixel;
	float sample_j = (float)(sample / cam.samples_per_pixel) / cam.samples_per_pixel;

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
    __shared__ Camera cam;
    __shared__ GuiState gui;
    cam = static_cast<Camera&&>(cam_g);
    gui = static_cast<GuiState&&>(gui_g);
    __syncthreads();
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
    if (idx >= cam.image_width*cam.image_height) return;
    unsigned int samples = cam.samples_per_pixel*cam.samples_per_pixel;

	pcg32_global = {rng_state[idx], 0};
// 	Camera cam2 = static_cast<Camera&&>(cam);


    if (gui.show_random) {
	    float3 rgb = gui.random_norm ? random_norm_float3() : make_float3(
	        ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32)
	    );
	    frame[idx] = to_pixel(make_float4_f3(rgb, 1.0f));
	    return;
    }

    // Ray Color
	int i = idx % cam.image_width;
	int j = idx / cam.image_width;
    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    for (int sample_idx = 0; sample_idx < samples; sample_idx++) {
        total = add(total, ray_color(get_ray(i, j, sample_idx, cam)));
        __syncthreads();
    }

    frame[idx] = to_pixel(make_float4_f3(div(total, samples), 1.0f));
}
