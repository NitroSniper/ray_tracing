typedef struct {
    float orig[3], dir[3];
} ray;


typedef union {
    struct
    {
        float r, g, b, a;
    } pixel;
    float channels[4];
} vec4;

typedef struct {
    float aspect_ratio;
    unsigned int image_width, image_height, samples_per_pixel;
    float center[3], pixel00_loc[3], pixel_delta[3];
} camera;

__device__ void clone(vec4 *const a, const vec4 const b) {
    memcpy(a, b, sizeof(a))
}

__device__ void v_add_v(vec4 *const a, const vec4 *const b, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] += b->channels[i];
    }
}

__device__ void v_add_s(vec4 *const a, const float s, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] += s;
    }
}

__device__ void v_sub_v(vec4 *const a, const vec4 *const b, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] -= b->channels[i];
    }
}

__device__ void v_sub_s(vec4 *const a, const float s, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] -= s;
    }
}

__device__ void v_mul_v(vec4 *const a, const vec4 *const b, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] *= b->channels[i];
    }
}

__device__ void v_mul_s(vec4 *const a, const float s, const int len) {
    for (int i = 0; i < len; i++) {
        a->channels[i] *= s;
    }
}


__device__ void ray_color(vec4 *const out, const ray const* r) {

}

// assumption can copy

extern "C" __global__ void render(vec4 *const frame, camera cam)
{
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
	if (idx >= cam.image_width*cam.image_height) return;

	// Happy path

	int i = idx % cam.image_width;
	int j = idx / cam.image_width;

	vec4 ray_center = {i, j, 0.0, 0.0};
	v_mul_v(ray_center, cam.pixel_delta, 3);
	v_add_v(ray_center, cam.pixel00_loc, 3);
	v_add_v(ray_center, cam.pixel00_loc, 3);
	v_sub_v(ray_center, cam.center, 3)




	frame[idx] = {0.0, (float)i / cam.image_width, (float)j / cam.image_height, 1.0};
	v_mul_s(&frame[idx], 255.0);
}

