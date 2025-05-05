


extern "C" __global__ void render(float* frames)
{
	unsigned int i = 1024*blockIdx.x + threadIdx.x;
	frames[i] = 100.0;
}

