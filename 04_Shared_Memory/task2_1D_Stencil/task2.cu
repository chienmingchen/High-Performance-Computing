// task1.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//

// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include "stencil.cuh"
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <string>


int main(int argc, char *argv[]) {

  float *image, *output, *mask;
  
  if (argc < 4) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  unsigned int imageSizeN = std::stoi(argv[1], nullptr, 0);
  unsigned int maskRadixR = std::stoi(argv[2], nullptr, 0);
  unsigned int threads_per_block = std::stoi(argv[3], nullptr, 0);
  // std::cout << imageSizeN << std::endl;

  /*
  // Alloc space for device copies of image, mask and output
  cudaMallocManaged(&image, imageSizeN * sizeof(float));
  cudaMallocManaged(&output, imageSizeN * sizeof(float));;
  cudaMallocManaged(&mask, (2 * maskRadixR + 1) * sizeof(float));

  for (int i = 0; i < N; i++) {
    image[i] = 1.0 - (float)std::rand() / ((RAND_MAX + 1u) / 2);
    out[i] = 1.0 - (float)std::rand() / ((RAND_MAX + 1u) / 2);
  }
  for (int i = 0 ; i< 2*)
  {
	mask[i] = 1.0 - (float)std::rand() / ((RAND_MAX + 1u) / 2);
  }
  
  */
  
  //Debug Pattern
  //imageSizeN = 7;
  //maskRadixR = 2;
  //threads_per_block = 10;
  
  std::cout << "n : " << imageSizeN << std::endl;
  std::cout << "R : " << maskRadixR << std::endl;
  std::cout << "thread : " << threads_per_block << std::endl;

  cudaMallocManaged(&image, imageSizeN * sizeof(float));
  cudaMallocManaged(&output, imageSizeN * sizeof(float));
  cudaMallocManaged(&mask, (2 * maskRadixR + 1) * sizeof(float));
  
  float imageX[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  float maskX[5] = {3.0, 4.0, 5.0, 4.0, 3.0};
    
  for(int i=0; i<7; i++)
    image[i] = imageX[i];
  for(int i=0; i<5; i++)
    mask[i] = maskX[i];

  stencil(image, mask, output, imageSizeN, maskRadixR, threads_per_block);

  cudaDeviceSynchronize();

  // Prints the ﬁrst element of the b scanned array
  //std::cout << output[0] << std::endl;

  // Prints the last element of the scanned array.
  //std::cout << output[N - 1] << std::endl;

  for(int i = 0; i < imageSizeN; i++)
	std::cout << output[i] << " ";
  std::cout << std::endl;

  // Cleanup
  cudaFree(image);
  cudaFree(output);
  cudaFree(mask);

  return 0;
}
