// task1.cpp : This file contains the 'main' function. Program execution begins
// and ends there.

// The std::chrono namespace provides timer functions in C++
#include <chrono>
// std::ratio provides easy conversions between metric units
#include <ratio>
// not needed for timers, provides std::pow function
#include <cmath>
// iostream is not needed for timers, but we need it for cout
#include "msort.h"
#include "omp.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
//#define check

int main(int argc, char *argv[]) {

  int *A;
  // int X[6] = {3, 5, 1, 2, 3, 1};

  if (argc < 2) {
    std::cerr << "error!! please check the argument" << std::endl;
    return 1;
  }

  std::size_t n = std::stoi(argv[1], nullptr, 0);
  std::size_t numThread = std::stoi(argv[2], nullptr, 0);
  std::size_t threshold = std::stoi(argv[3], nullptr, 0);

  //std::cout << "numThread : " << numThread << std::endl;
  A = new (std::nothrow) int[n];
  if (!A) {
    std::cout << "Memory allocation failed\n";
    return 0;
  }

  // Creates an array of n random double
  for (std::size_t i = 0; i < n; i++) {
    A[i] = std::rand() / (RAND_MAX / 31);
    // std::cout << A[i] << " ";
  }
  // std::cout << std::endl;
  omp_set_num_threads(numThread);
  // Get the starting timestamp
  double start = omp_get_wtime();
  // run msort parallelly with t threads
//#pragma omp parallel
  //{
//#pragma omp single
  msort(A, n, threshold);
  //}
  // Get the ending timestamp
  double end = omp_get_wtime();

  // Prints the ﬁrst element of the output scanned array
  std::cout << A[0] << std::endl;
  // Prints the last element of the scanned array.
  std::cout << A[n - 1] << std::endl;

  std::cout << (end - start) * 1000 << "\n";

#ifdef check
  for (std::size_t i = 0; i < n; i++) {
    std::cout << A[i] << " ";
  }
  std::cout << std::endl;
#endif

  delete[] A;

  return 0;
}
