extern "C" __global__ void addKernel(int* C, const int* A, int* B)
{
	unsigned int i = threadIdx.x;
	C[i] = A[i] + B[i];

}

