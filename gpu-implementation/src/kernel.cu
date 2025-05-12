typedef union {
    struct
    {
        float r, g, b, a;
    } pixel;
    float channels[4];
} pixel;

typedef struct {
    float aspect_ratio;
    unsigned int image_width, image_height, samples_per_pixel;
    float center[3], pixel00_loc[3], pixel_delta[3];
} camera;

__device__ void multiplyBy255(pixel* p) {
    for (int i = 0; i < 4; i++) {
        p->channels[i] *= 255;
    }
}

extern "C" __global__ void render(pixel* frame, camera cam)
{
	unsigned int i = 1024*blockIdx.x + threadIdx.x;
	if (i >= cam.image_width*cam.image_height) return;

	// Happy path

	int x = i % cam.image_width;
	int y = i / cam.image_width;
	frame[i] = {0.0, (float)x / cam.image_width, (float)y / cam.image_height, 1.0};
	multiplyBy255(&frame[i]);
}

