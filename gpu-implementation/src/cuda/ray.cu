__device__ LightRecord hit_world(Ray& r, float2 t, Sphere* world, const unsigned int world_size) {
    HitRecord record;
    LightRecord best;
    float closest_t_max = t.y;
    for (int i = 0; i < world_size; i++) {
        Sphere& s = world[i];
        record = s.hit(r, make_float2(t.x, closest_t_max));
        if (!record.is_none) {
            closest_t_max = record.t;
            Light light = s.mat.scatter(r, record);
            best = LightRecord(
                static_cast<HitRecord&&>(record),
                static_cast<Light&&>(light)
            );
        }
    }
    return best;
}



__device__ float3 ray_color(Ray r, Sphere* world, const unsigned int world_size, unsigned int max_depth) {
    bool bounce;
    float2 t = make_float2(0.001f, 10024.0f);
    float3 color = make_float3(1.0f, 1.0f, 1.0f);

    float3 nothing = make_float3(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth <= max_depth; depth++) {
        if (depth == max_depth) {
            return nothing;
        }
        bounce = false;
        LightRecord lr = hit_world(r, t, world, world_size);
        if (!lr.record.is_none) {
            r = static_cast<Ray&&>(lr.light.ray);
            if (lr.light.is_none) color = mul(color, lr.light.color);
            else return nothing;
            bounce = true;
        }
        if (bounce) continue;

        float3 unit_dir = normalize(r.dir);
        float a = (unit_dir.y + 1.0f) * 0.5f;
        color = mul(color, add(mul(make_float3(1.0f, 1.0f, 1.0f), 1.0f-a), mul(make_float3(0.5f, 0.7f, 1.0f), a)));
        break;
    }
        return color;
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

	float3 pixel_center = add(
	    cam.pixel00_loc,
	    add(
	        mul(
	            cam.pixel_delta_u,
	            (float)i + sample_i
	        ),
	        mul(
	            cam.pixel_delta_v,
	            (float)j + sample_j
	        )
	    )
	);

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


extern "C" __global__ void render(uint64_t *rng_state, uchar4 *const frame, Camera cam, GuiState gui) {
    // Load shared data
    const unsigned int world_size = 9;
    Sphere world[world_size];

    world[0] = Sphere(make_float3(0.0f, -100.5f,-1.0f), 100.0f, Diffuse(make_float3(0.8f,0.8f,0.0f)));
    world[1] = Sphere(make_float3(0.0f,0.0f,-1.2f), 0.5f, Diffuse(make_float3(0.1f,0.2f,0.5f)));
    world[2] = Sphere(make_float3(-1.0,0.0f,-1.2f), 0.5f, Reflect(make_float3(0.8f,0.8f,0.8f), 0.3f));
    world[3] = Sphere(make_float3(1.0,0.0f,-1.2f), 0.5f, Reflect(make_float3(0.8f,0.6f,0.2f), 1.0f));
    world[4] = Sphere(make_float3(1.0, 0.0f, -3.2f), 0.5f, Reflect(make_float3(0.8f,0.6f,0.2f), 0.0f));
    // Reflection
    world[5] = Sphere(make_float3(0.0, 0.0f, -50.0f), 25.0f, Reflect(make_float3(0.9f,0.9f,0.9f), 0.0f));
    world[6] = Sphere(make_float3(0.0, 0.0f, -110.0f), 25.0f, Reflect(make_float3(0.9f,0.9f,0.9f), 0.0f));
    world[7] = Sphere(make_float3(2.0,0.0f, -80.0f), 0.5f, Diffuse(make_float3(0.8f,0.6f,0.2f)));
    world[8] = Sphere(make_float3(-2.0,0.0f, -80.0f), 0.5f, Reflect(make_float3(0.8f,0.6f,0.2f), 0.3f));

    // if thread doesn't have a pixel
	unsigned int idx = gui.block_dim*blockIdx.x + threadIdx.x;
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
        total = add(total, ray_color(get_ray(i, j, sample_idx, gui.sample2_per_pixel, cam), world, world_size, gui.max_depth));
    }

    frame[idx] = to_pixel(make_float4_f3(sqrt(div(total, samples)), 1.0f));
}
