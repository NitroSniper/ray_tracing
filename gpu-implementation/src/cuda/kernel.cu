__device__ float3 ray_color(const ray r) {
    sphere s = {make_float3(0.0,0.0,-1.0), 0.5f};
    hit_record record = hit_sphere(s, r, make_float2(0.001, 1024.0));
    if (!record.is_none) {
        return make_float3(1.0, 0.0,0.0);
    }

    float3 unit_dir = mul(r.dir, rnorm3df(r.dir.x, r.dir.y, r.dir.z));
    float a = (unit_dir.y + 1.0f) * 0.5f;
    return add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a ));
}

extern "C" __global__ void render(float4 *const frame, const camera cam)
{
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
	if (idx >= cam.image_width*cam.image_height) return;

	// Happy path

	int i = idx % cam.image_width;
	int j = idx / cam.image_width;

    float3 pixel_center = add(mul(make_float3((float)i, (float)j, 0.0), cam.pixel_delta), cam.pixel00_loc);
    ray r = {cam.center, sub(pixel_center, cam.center)};



	// frame[idx] = make_float4(0.0f, (float)i / cam.image_width, (float)j / cam.image_height, 1.0);
	frame[idx] = make_float4_f3(ray_color(r), 1.0);
	frame[idx] = mul(frame[idx], 255.0f);
}

