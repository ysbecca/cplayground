
#include <stdio.h>
#include "check.h"
#define N 10

__global__
void add(int *a, int *b, int *c)
{
  int tid = blockIdx.x;
  if (tid < N) {
      c[tid] = a[tid] + b[tid];
  }
}

int main(int argc, char **argv)
{
  int A[N], B[N], C[N];

  int *dev_A, *dev_B, *dev_C;

  // Allocate arrays on the GPU
  CHECK( cudaMalloc( (void**)&dev_A, N*sizeof(int)) );
  CHECK( cudaMalloc( (void**)&dev_B, N*sizeof(int)) );
  CHECK( cudaMalloc( (void**)&dev_C, N*sizeof(int)) );

  // Initialise the arrays on the CPU
  for (int i = 0; i < N; i++) {
      A[i] = -i;
      B[i] = i*i;
  }

  // Copy the input arrays from host to device
  CHECK( cudaMemcpy( dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK( cudaMemcpy( dev_B, B, N*sizeof(int), cudaMemcpyHostToDevice) );

  // Perform addition via the kernel

  add<<<N,1>>>( dev_A, dev_B, dev_C );

  // Copy the results back from device to host array
  CHECK( cudaMemcpy( C, dev_C, N*sizeof(int), cudaMemcpyDeviceToHost) );

  // Output result
  for (int i = 0; i < N; i++)
    printf( "%d + %d = %d\n", A[i], B[i], C[i]);

  // Clean up device.
  cudaFree( dev_A );
  cudaFree( dev_B );
  cudaFree( dev_C );

  return 0;
}
