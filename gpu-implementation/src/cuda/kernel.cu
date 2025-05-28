struct material_record {
    hit_record rec;
    material mat;
};

__device__ material_record hit_world(ray r, float2 t, sphere* world, int world_size) {
    hit_record temp_rec;
    hit_record record;
    material mat = diffuse(make_float3(1.0f, 1.0f, 1.0f));
    float closest_t_max = t.y;
    for (int i = 0; i < world_size; i++) {
        sphere s = world[i];
        temp_rec = s.hit(r, make_float2(t.x, closest_t_max));
        if (!temp_rec.is_none) {
            mat = (material) s.mat;
            closest_t_max = temp_rec.t;
            record = temp_rec;
        }
    }
    return {record, mat};
}


__device__ float3 ray_color(ray r, sphere* world, int world_size) {
//     sphere s = {make_float3(0.0,0.0,-1.0), 0.5f};
//     hit_record record = s.hit(r, make_float2(0.001, 1024.0));
    float3 color = make_float3(1.0f, 1.0f, 1.0f);

    for (int depth = 10; depth >= 0; depth--) {
        bool bounce = false;
        material_record record = hit_world(r, {0.001f, 1024.0f}, world, world_size);
        if (!record.rec.is_none) {
            light l = record.mat.scatter(r, record.rec);
            r = l.ray;
            color = mul(color, l.color);
//             bounce = true;
        }
        if (bounce) continue;
        float3 unit_dir = normalize(r.dir);
        float a = (unit_dir.y + 1.0f) * 0.5f;
        color = mul(color, add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a)));
        break;
    }
    return color;
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

    material material_ground = diffuse({0.8f, 0.8f, 0.0f});
    material material_center = diffuse({0.1f, 0.2f, 0.5f});
    material material_left   = reflect({0.8f, 0.8f, 0.8f});
    material material_right  = reflect({0.8f, 0.6f, 0.2f});
    sphere s =    sphere(make_float3( 0.0f, -100.5f, -1.0f), 100.0, material_ground);
    sphere foo =  sphere(make_float3( 0.0f,    0.0f, -1.2f),   0.5, material_center);
    sphere fizz = sphere(make_float3(-1.0f,    0.0f, -1.0f),   0.5, material_left);
    sphere bazz = sphere(make_float3( 1.0f,    0.0f, -1.0f),   0.5, material_right);
    const int world_size = 4;
    sphere world[world_size] = { s, foo, fizz, bazz };
//     hittable(sphere(make_float3(-1.0f,    0.0f, -1.0f),   0.5), material_left  ),
//     hittable(sphere(make_float3( 1.0f,    0.0f, -1.0f),   0.5), material_right )};


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
//     float3 rgb = {0.0f, 0.0f, 0.0f};
//     hit_record r = h.hit(get_ray(i, j, 0, cam), {0.0f, 100024.0f});
//     if (!r.is_none) {
//         rgb = {1.0f, 1.0f, 0.0f};
//     }
    for (int sample_idx = 0; sample_idx < cam.samples_per_pixel*cam.samples_per_pixel; sample_idx++) {
        total = add(total, ray_color(get_ray(i, j, sample_idx, cam), world, world_size));
    }



// 	frame[idx] = make_float4(0.0f, (float)i / cam.image_width, (float)j / cam.image_height, 1.0);
    float3 rgb = div(total, cam.samples_per_pixel*cam.samples_per_pixel);
    frame[idx] = to_pixel(make_float4_f3(rgb, 1.0f));

}

