#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include <cstdlib>
#include <cstring>
#include <iomanip>

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(-10,10);
  return dist(rng);
}


int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  unsigned int n = std::stoi(argv[1], nullptr, 0);
  
  // generate n random numbers on the host
  thrust::host_vector<float> h_vec(n);

  //thrust::generate(h_vec.begin(), h_vec.end(), rand);
  thrust::fill(h_vec.begin(), h_vec.end(), 1);
 

  // transfer data to the device
  thrust::device_vector<float> d_vec = h_vec;
  //thrust::device_vector<float> d_vec2[n];
  //thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
  
  /*
  for(int i=0; i<n; i++)
	std::cout << h_vec[i] << " ";
  std::cout << std::endl;  
  */

  //int init = d_vec[0];

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // run reduce on the device
  //thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin(), 0,  thrust::plus<int>());
  thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin());
  
  //cudaDeviceSynchronize(); //i wait until prior kernel is finished

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << ms << "ms" << std::endl;

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  cudaDeviceSynchronize();

  for(unsigned int i=n-100; i<n; i++)
  std::cout <<  std::setprecision(12)  << h_vec[i] << std::endl;

  return 0;
}
