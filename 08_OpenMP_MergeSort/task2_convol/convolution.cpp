#include <iostream>
#include "convolution.h"
#include <omp.h>

void Convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {

  std::size_t offset = (m - 1) / 2;
  std::size_t x, y, i, j;

// shared(image,mask,output,offset,m,n)
#pragma omp parallel
  {
#pragma omp for private(x, y, i, j)
    for (x = 0; x < n; x++) {
      for (y = 0; y < n; y++) {
        output[x * n + y] = 0;
        for (i = 0; i < m; i++) {
          for (j = 0; j < m; j++) {

            if ((x + i - offset) >= 0 && (y + j - offset) >= 0 &&
                (x + i - offset) < n && (y + j - offset) < n) {
              output[x * n + y] +=
                  mask[m * i + j] *
                  image[(x + i - offset) * n + (y + j - offset)];
            }
          }
        }
        // int threadID = omp_get_thread_num();
        // printf("threadID = %d  temp = %f\n", threadID, temp);
        // output[x * n + y] = temp;
      }
    }
  }
}
