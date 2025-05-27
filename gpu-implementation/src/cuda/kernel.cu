__device__ float3 ray_color(const ray r) {
    sphere s = {make_float3(0.0,0.0,-1.0), 0.5f};
    hit_record record = s.hit(r, make_float2(0.001, 1024.0));
    if (!record.is_none) {
        float3 n = normalize(sub(ray_at(r, record.t), make_float3(0.0, 0.0, -1.0)));
        return mul(make_float3(n.x+1.0, n.y+1.0,n.z+1.0), 0.5);

    }

    float3 unit_dir = normalize(r.dir);
    float a = (unit_dir.y + 1.0f) * 0.5f;
    return add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a ));
}

__device__ ray get_ray(
    const unsigned int i,
    const unsigned int j,
    const unsigned int sample,
    const camera cam
) {
	float sample_i = (float)(sample % cam.samples_per_pixel) / cam.samples_per_pixel;
	float sample_j = (float)(sample / cam.samples_per_pixel) / cam.samples_per_pixel;

//     if(threadIdx.x==0){
//         printf("i=%f \n", sample_i);
//         printf("j=%f \n", sample_j);
//     }

    float3 pixel_center = add(mul(make_float3((float)i+sample_i, (float)j+sample_j, 0.0), cam.pixel_delta), cam.pixel00_loc);
    ray r = {cam.center, sub(pixel_center, cam.center)};
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

typedef struct {
    bool show_random;
    bool random_norm;
} gui_state;

extern "C" __global__ void render(uint64_t *rng_state, uchar4 *const frame, const camera cam, const gui_state gui) {
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
	pcg32_global = {rng_state[idx], 0};
    if (idx >= cam.image_width*cam.image_height) return;

    if (gui.show_random) {

	    float3 rgb = gui.random_norm ? random_norm_float3() : make_float3(
	        ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32),
            ldexpf(pcg32_random_r(), -32)
	    );
	    frame[idx] = to_pixel(make_float4_f3(rgb, 1.0f));
	    return;
    }

	// Happy path

	int i = idx % cam.image_width;
	int j = idx / cam.image_width;

    float3 total = make_float3(0.0,0.0,0.0);
    for (int sample_idx = 0; sample_idx < cam.samples_per_pixel*cam.samples_per_pixel; sample_idx++) {
        total = add(total, ray_color(get_ray(i, j, sample_idx, cam)));
    }



	// frame[idx] = make_float4(0.0f, (float)i / cam.image_width, (float)j / cam.image_height, 1.0);
    float3 rgb = div(total, cam.samples_per_pixel*cam.samples_per_pixel);
    frame[idx] = to_pixel(make_float4_f3(rgb, 1.0f));

}

