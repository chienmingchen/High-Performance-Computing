#include <iostream>
#include <omp.h>
#include "matmul.h"

void mmul(const float *A, const float *B, float *C, const std::size_t n) {

  std::size_t i,j,k;
  # pragma omp parallel shared(A, B, C) private(i, j, k)
  {
    # pragma omp for
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        C[i * n + j] = 0;
        for (k = 0; k < n; k++) {
          C[i * n + j] += A[i * n + k] * B[j * n + k];
        }
      }
    }
  }
}

