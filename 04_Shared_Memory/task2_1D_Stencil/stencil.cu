
#include "stencil.cuh"
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <string>



__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {



   // The element of the block sub-matrix that is computed
   // by the thread
   
    // Shared memory for the mask
	// __shared__ variables cannot have an initialization as part of their declaration.
	extern __shared__ float s[];
	
	float *filterD = s; 
	float *inD = (float*)&filterD[(2*R) + 1];
	float *outD = (float*)&inD[blockDim.x + (2*R)];

	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + R;
	
	// Load tiles from global memory into s:hared memory; each
    // thread loads one element of the two tiles from image & mask
	
	// 1.Ensure that it won't load image exceeding the boundary 
	inD[lindex] = (gindex < n)? image[gindex] : 0;   //inD[threadIdx.boundary + R] = image[n-1], inD[threadIdx.over + R] = 0 
	//std::printf("inD[%d] : %f\n", lindex, inD[lindex]);

	// 2.Padding for the temp image buffer inD
	if (threadIdx.x < R) {
		//pad from previous tile region or 0
		inD[threadIdx.x] = (gindex > R)? image[gindex - R] : 0;
		inD[lindex + blockDim.x] = ((gindex + blockDim.x) < n)? image[gindex + blockDim.x] : 0;
		//std::printf("Padding : inD[%d] : %f\n", threadIdx.x, inD[threadIdx.x]);
	}
	//if (lindex >= blockDim.x && gindex < n){
		//pad from next tile region or 0
	//	inD[lindex] = ((gindex+R) < n) ? image[gindex + R] : 0;	
	//	std::printf("Padding : inD[%d] : %f\n", lindex, inD[lindex] );
	//}

	//std::printf("%f ", inD[lindex]);
	
	// 3.Ensure mask range within the boundary 2R+1
	if(threadIdx.x < (2*R+1) ){
		filterD[threadIdx.x] = mask[threadIdx.x];
		//std::printf("filter[%d] : %f\n", threadIdx.x, filterD[threadIdx.x] );
	}


	// Synchronize (ensure all the data is available)
        __syncthreads();
	
	std::printf("blok(%d),lindex(%d) : %f\n", blockIdx.x, lindex, inD[lindex] );

	//std::printf("block: %d, thread: %d, inD : %f, mask : %f\n", blockIdx.x, threadIdx.x, inD[threadIdx.x], filterD[threadIdx.x]);

	
	//After shared memory data is synchronized, it can do the math'
	if(gindex < n){
	    for (int k = 0; k < (2*R+1); k++){
		outD[threadIdx.x] += inD[threadIdx.x + k] * filterD[k];
		if(threadIdx.x == 4)
		std::printf("%d : %f x %f \n",gindex, inD[threadIdx.x + k], filterD[k]);
	    }
	    output[gindex] = outD[threadIdx.x];
	}
	// Synchronize before global memory access
        //__syncthreads();
	
	// Once the job is done, move the result from shared memory back to the global memory
	//output[gindex] = outD[threadIdx.x];
		
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block){

	int blockNum = (n + threads_per_block -1)/threads_per_block;
	
	std::cout << "blockNum : " << blockNum << std::endl;
	std::cout << "threads_per_block" << threads_per_block << std::endl;
	
	// Launch the device computation
	stencil_kernel<<<blockNum, threads_per_block,(2*threads_per_block + 4*R + 1)*sizeof(float)>>>(image, mask, output, n, R);
 
}
