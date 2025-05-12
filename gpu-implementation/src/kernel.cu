typedef struct {
    float orig[3], dir[3];
} ray;

typedef struct {
    float aspect_ratio;
    unsigned int image_width, image_height, samples_per_pixel;
    float3 center, pixel00_loc, pixel_delta;
} camera;

__device__ float4 mul(const float4 a, const float s) {
  return make_float4(a.x*s, a.y*s, a.z*s, a.w*s);
}

extern "C" __global__ void render(float4 *const frame, const camera cam)
{
	unsigned int idx = 1024*blockIdx.x + threadIdx.x;
	if (idx >= cam.image_width*cam.image_height) return;

	// Happy path

	int i = idx % cam.image_width;
	int j = idx / cam.image_width;

	frame[idx] = make_float4(0.0, (float)i / cam.image_width, (float)j / cam.image_height, 1.0);
	frame[idx] = mul(frame[idx], 255.0f);
}

