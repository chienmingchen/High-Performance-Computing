
#include "vadd.cuh"
#include <cuda.h>


__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int row = index / n;
	int col = index % n;

	if(row < n && column < n)
	{
		float temp = 0;
		for (int k = 0; k < n; k++) {
		temp += A[row * n + k] * B[k * n + col];
		}
		C[row * n + col] = temp
	}
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
   
	int size = n * n * sizeof(float);
	int blockNum = (size + threads_per_block -1)/threads_per_block;
   
	// Load A and B to the device
	float* dA;
	cudaMalloc((void**)&dA, size);
	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

	float* dB;
	size = n * n * sizeof(float);
	cudaMalloc((void**)&dB, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* dC;
	cudaMalloc((void**)&dC, size);
  
	// Launch the device computation
	matmul_kernel<<<blockNum, threads_per_block>>>(dA,dB,dC,n);
 
	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
  
	// Free device memory
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}