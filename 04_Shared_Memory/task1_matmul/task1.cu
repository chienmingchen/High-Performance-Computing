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
#include "vadd.cuh"
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <iostream>
#include <string>

// Provide some namespace shortcuts
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Set some limits for the test
// const size_t TEST_SIZE = 1000;
// const size_t TEST_MAX = 32;

#define THREADS_PER_BLOCK 512

int main(int argc, char *argv[]) {
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  float *a, *b; // host copies of a, b, c
  // float *d_a, *d_b; // device coipies of a, b, c

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  int N = std::stoi(argv[1], nullptr, 0);
  // std::cout << N << std::endl;

  int size = N * sizeof(float);
  // Alloc space for device copies of a, b
  // cudaMallocManaged((void **)&d_a, size);
  // cudaMallocManaged((void **)&d_b, size);

  cudaMallocManaged(&a, N * sizeof(float));
  cudaMallocManaged(&b, N * sizeof(float));

  // Alloc space for host copies of a, b, c and setup input values
  // a = new(std::nothrow) float[N];
  // b = new(std::nothrow) float[N];

  for (int i = 0; i < N; i++) {
    a[i] = 1.0 - (float)std::rand() / ((RAND_MAX + 1u) / 2);
    b[i] = 1.0 - (float)std::rand() / ((RAND_MAX + 1u) / 2);
  }

  // Get the starting timestamp
  start = high_resolution_clock::now();

  // Copy inputs to device
  // cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU
  vadd<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, b, size);

  cudaDeviceSynchronize();

  // Copy result back to host
  // cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  // Get the ending timestamp
  end = high_resolution_clock::now();

  // Convert the calculated duration to a double using the standard library
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  // Durations are converted to milliseconds already thanks to
  // std::chrono::duration_cast
  std::cout << "Total time: " << duration_sec.count() << "ms\n";

  // Prints the ﬁrst element of the b scanned array
  std::cout << b[0] << std::endl;

  // Prints the last element of the scanned array.
  std::cout << b[N - 1] << std::endl;

  // Cleanup
  // delete[] a; delete[] b;
  cudaFree(a);
  cudaFree(b);

  return 0;
}
