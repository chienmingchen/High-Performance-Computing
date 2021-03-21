#include <iostream>
#include "matmul.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

void msort_serial(int *arr, std::size_t size);
void msort_parallel(int *arr, const std::size_t n, const std::size_t threshold);

void msort(int *arr, const std::size_t n, const std::size_t threshold) {

#pragma omp parallel
  {
#pragma omp single
    msort_parallel(arr, n, threshold);
  }
}

void msort_parallel(int *arr, const std::size_t n,
                    const std::size_t threshold) {
  if(n < 2) return;

  if (n < threshold) {
    msort_serial(arr, n);
    return;
  }

#pragma omp task firstprivate(arr, n, threshold)
  msort_parallel(arr, n / 2, threshold);

#pragma omp task firstprivate(arr, n, threshold)
  msort_parallel(arr + n / 2, n - n / 2, threshold);

#pragma omp taskwait
  // merge(arr, n);
  std::inplace_merge(arr, arr + n / 2, arr + n);
}

void msort_serial(int *arr, std::size_t n) {

  int key, j;

  for (std::size_t i = 1; i < n; i++) {
    key = arr[i];
    j = i - 1;

    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;
  }
}

/*
void merge(int *X, std::size_t n) {
  std::size_t i = 0;
  std::size_t j = n / 2;
  std::size_t ti = 0;

  int *tmp;
  tmp = new (std::nothrow) int[n];
  if (!tmp) {
    std::cout << "Memory allocation
failed\n"; return;
  }

  while (i < n / 2 && j < n) {
    if (X[i] < X[j]) {
      tmp[ti] = X[i];
      ti++;
      i++;
    } else {
      tmp[ti] = X[j];
      ti++;
      j++;
    }
  }
  while (i < n / 2) {
    tmp[ti] = X[i];
    ti++;
    i++;
  }
  while (j < n) {
    tmp[ti] = X[j];
    ti++;
    j++;
  }
  memcpy(X, tmp, n * sizeof(int));
}
*/
