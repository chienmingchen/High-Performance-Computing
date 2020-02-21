
#include "stencil.cuh"
#include <cuda.h>


__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {


   //Muld(float* A, float* B, int wA, int wB, float* C)

   // Block index
   int bx = blockIdx.x * blockDim.x;

   // Thread index
   int tx = threadIdx.x;

   // The element of the block sub-matrix that is computed
   // by the thread
   float Csub = 0;
   
    // Shared memory for the mask
	// __shared__ variables cannot have an initialization as part of their declaration.
	__shared__ float filterD[2 * R + 1];
	__shared__ float inD[blockDim.x + (2 * R)];   //inD[] = preImg[last R] + Img[n] + nextImg[first R]
	__shared__ float outD[blockDim.x];
	
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + R;
	
	// Load tiles from global memory into shared memory; each
    // thread loads one element of the two tiles from image & mask
	
	// 1.Ensure that it won't load image exceeding the boundary 
	inD[lindex] = (gindex < n)? image[gindex] : 0;   //inD[threadIdx.boundary + R] = image[n-1], inD[threadIdx.over + R] = 0 
    
	// 2.Padding for the temp image buffer inD
	if (threadIdx.x < R && gindex < n) {
	
		//pad from previous tile region or 0
		inD[threadIdx.x] = (gindex > R)? image[gindex - R + threadIdx.x] : 0;
		
		//pad from next tile region or 0
		inD[threadIdx.x + blockDim.x] = (gindex < (n - R)) ? image[gindex + threadIdx.x] : 0;
	}
	
	// 3.Ensure mask range within the boundary 2R+1 
	filterD[threadIdx.x] = (threadIdx.x < (2 * R + 1))? mask[threadIdx.x] : 0 ;
  
	// Synchronize (ensure all the data is available)
    __syncthreads();

	std::printf("block: %d, thread: %d, inD : %d, mask : %d\n", blockIdx.x, threadIdx.x, inD[threadIdx.x], filterD[threadIdx.x]);

	/*
	// After shared memory data is synchronized, it can do the math
	for (int k = 0; k < (2*R+1); k++)
	outD[threadIdx.x] += inD[threadIdx.x + k] * filterD[threadIdx.x + k];
	
	// Synchronize before global memory access
    __syncthreads();
	*/
	
	// Once the job is done, move the result from shared memory back to the global memory
	output[gindex] = outD[threadIdx];
		
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){

	int blockNum = (n + threads_per_block -1)/threads_per_block;
	
	std::cout << "blockNum : " << blockNum << std::endl;
	
	// Launch the device computation
	stencil_kernel<<<blockNum, threads_per_block>>>(image, mask, output, n, R);
 
}