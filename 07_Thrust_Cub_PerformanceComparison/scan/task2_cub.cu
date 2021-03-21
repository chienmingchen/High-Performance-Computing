#define CUB_STDERR // print CUDA runtime errors to console
#include <iostream>
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>
// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include <cstdlib>
#include <cstring>

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "error!! please check the argument" << std::endl;
        return 1;
    }

    unsigned int num_items = std::stoi(argv[1], nullptr, 0);
  
    // Set up host arrays
    float* h_in = new(std::nothrow) int[num_items];
	
	for (unsigned int i = 0; i < num_items; i++) {
        h_in[i] = 1.0 - (float)(2*std::rand()/RAND_MAX);
    }
  
    float  sum = 0;
    for (unsigned int i = 0; i < num_items-1; i++)
        sum += h_in[i];

    // Set up device arrays
    int* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * num_items));
	
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
	
    // Setup device output array
    int* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * num_items));
	
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
  
    // Do the actual reduce operation
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
	
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
	
	  
    int gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, (d_sum+num_items-1), sizeof(float) * 1, cudaMemcpyDeviceToHost));
	
    // Check for correctness
	//if(gpu_sum != sum)
	//{
		printf("\t%s\n", (gpu_sum == sum ? "Test passed." : "Test falied."));
		printf("\tSum is: %d, Expected: %d\n", gpu_sum, sum);
	//}
	
	std::cout << gpu_sum << std::endl;
	std::cout << ms << "ms" << std::endl;
	

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}
