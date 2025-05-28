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
    frame[idx] = to_pixel(make_float4(0.0f, (float)i / cam.image_width, (float)j / cam.image_height, 1.0));
}
