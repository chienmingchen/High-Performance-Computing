#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include <cstdlib>
#include <cstring>

int main(void) {

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  unsigned int n = std::stoi(argv[1], nullptr, 0);
  
  // generate n random numbers on the host
  thrust::host_vector<int> h_vec(n);
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  int init = X[0];

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  
  // run reduce on the device
  double result = thrust::reduce(X.begin(), X.end(), init,  thrust::plus<int>());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << ms << "ms" << std::endl;
  
  std::cout << "maximum is " << result << "\n";

  return 0;
}
