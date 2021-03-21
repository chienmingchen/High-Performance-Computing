#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include "count.cuh"

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts) {

  //retrieve the input length
  int inlen = din.size();
  std::cout << "input lenght : " << inlen << std::endl;  

  //prepare the key table by copying from din
  thrust::device_vector<int> d_vec = d_in;
  
  //prepare the values table with all 1
  thrust::device_vector<int> d_count(inlen);
  thrust::fill(d_count.begin(), d_count.end(), 1);

  //step1 sort the key table
  thrust::sort(d_vec.begin(), d_vec.end());
  bool result = thrust::is_sorted(d_vec.begin(), d_vec.end());

  //step2 find jumps
  int outlen = thrust::inner_product(d_vec.begin(), d_vec.end() - 1, d_vec.begin() + 1, 0, thrust::plus<int>(), thrust::not_equal_to<int>()) + 1;
  std::cout << "jumps : " << outlen << std::endl;
  
  //step3 resize the output array
  values.resize(inlen);
  counts.resize(inlen);

  //run the function reduce_by_key to perfome the count behavior
  thrust::reduce_by_key(d_vec.begin(), d_vec.end(), d_count.begin(), value.begin(), count.begin());

  //copy the result back 
  //thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
