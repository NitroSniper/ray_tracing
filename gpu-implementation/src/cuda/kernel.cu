__device__ float3 ray_color(const ray r) {
    sphere s = {make_float3(0.0,0.0,-1.0), 0.5f};
    hit_record record = hit_sphere(s, r, make_float2(0.001, 1024.0));
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

extern "C" __global__ void render(float4 *const frame, const camera cam) {
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
	if (idx >= cam.image_width*cam.image_height) return;

	// Happy path

	int i = idx % cam.image_width;
	int j = idx / cam.image_width;

    float3 total = make_float3(0.0,0.0,0.0);
    for (int sample_idx = 0; sample_idx < cam.samples_per_pixel*cam.samples_per_pixel; sample_idx++) {
        total = add(total, ray_color(get_ray(i, j, sample_idx, cam)));
    }



	// frame[idx] = make_float4(0.0f, (float)i / cam.image_width, (float)j / cam.image_height, 1.0);
	frame[idx] = make_float4_f3(div(total, cam.samples_per_pixel*cam.samples_per_pixel), 1.0);
	frame[idx] = mul(frame[idx], 255.0f);
}

