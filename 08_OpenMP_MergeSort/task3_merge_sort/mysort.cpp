#include <iostream>
#include "matmul.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

void merge(int *X, std::size_t n);
void msort(int *arr, const std::size_t n, const std::size_t threshold);

int main(){
   int *arr = new int[10];
   for(int i=0; i<10 ; i++){
      arr[i]=10-i;
      std::cout << arr[i] << " ";
   }
   std::cout << std::endl;

   msort(arr, 10, 5);

   for(int i=0; i<10 ; i++){
      std::cout << arr[i] << " ";
   }

}


void msort(int *arr, const std::size_t n, const std::size_t threshold) {

  
  if(n < 2) return;

  //you_need_add_another_sort_here(arr, n);

#pragma omp task firstprivate(arr, n, threshold)
  msort(arr, (n/2), threshold);

#pragma omp task firstprivate(arr, n, threshold)
  msort(arr + (n/2), n - (n/2), threshold);

#pragma omp taskwait
  merge(arr, n);
}



void merge(int *X, std::size_t n) {
  std::size_t i = 0;
  std::size_t j = n / 2;
  std::size_t ti = 0;

  int *tmp = new (std::nothrow) int[n];

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
