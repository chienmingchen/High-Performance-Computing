#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
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

  int n = std::stoi(argv[1], nullptr, 0);
  
  //generate n random numbers on the host
  thrust::host_vector<int> h_vec(n);
  //thrust::generate(h_vec.begin(), h_vec.end(), rand);
  
  //test pattern
  thrust::host_vector<int> X(6);
  X[0]=3; X[1]=5; X[2]=1; X[3]=2; X[4]=3; X[5]=1;

  //transfer data to the device
  thrust::device_vector<int> d_in = X;
  thrust::device_vector<int> value;
  thrust::device_vector<int> count;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  //run count
  count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts)

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  //Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << ms << "ms" << std::endl;
  
  //AS TA's comments, print the result directly without copying them back to the host
  int outlen = value.size();
  
  for(int i=0; i<inlen; i++)
    std::cout << h_vec[i] << " ";
  std::cout << std::endl;

  for(int i=0; i<outlen; i++)
    std::cout << value[i] << " ";
  std::cout << std::endl;

  for(int i=0; i<outlen; i++)
    std::cout << count[i] << " ";
  std::cout << std::endl;


  return 0;
}
