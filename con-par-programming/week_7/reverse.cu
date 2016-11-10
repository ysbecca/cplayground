
#include <stdio.h>
#include "check.h"
#define N 10

__global__
void reverse(int *A, int *R)
{
  int tid = blockIdx.x;
  if (tid < N) {
      R[N - tid - 1] = A[tid];
  }
}

int main(int argc, char **argv)
{
  int A[N], R[N];

  int *dev_A, *dev_R;

  // Allocate arrays on the GPU
  CHECK( cudaMalloc( (void**)&dev_A, N*sizeof(int)) );
  CHECK( cudaMalloc( (void**)&dev_R, N*sizeof(int)) );

  // Initialise the arrays on the CPU
  for (int i = 0; i < N; i++) {
      A[i] = i;
  }

  // Copy the input arrays from host to device
  CHECK( cudaMemcpy( dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK( cudaMemcpy( dev_R, R, N*sizeof(int), cudaMemcpyHostToDevice) );

  // Perform addition via the kernel
  reverse<<<N,1>>>( dev_A, dev_R );

  // Copy the results back from device to host array
  CHECK( cudaMemcpy( R, dev_R, N*sizeof(int), cudaMemcpyDeviceToHost) );

  // Output result
  for (int i = 0; i < N; i++)
    printf( "%d, %d\n", A[i], R[i]);

  // Clean up device.
  cudaFree( dev_A );
  cudaFree( dev_R );

  return 0;
}
