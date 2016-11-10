
#include <stdio.h>
#include "check.h"
#define N 32 // Matrix dimension
#define MAX_THREADS 1024 //1024

__global__
void transpose(int *M, int *T)
{
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;
  int tid1 = (gridDim.x * b_y) + b_x;
  int tid2 = (gridDim.x * b_x) + b_y;
  if (b_x < N && b_y < N) {
      T[tid1] = M[tid2];
  }
}

int main(int argc, char **argv)
{
  int M[N][N], T[N][N];
  int *dev_M, *dev_T;

  // Allocate arrays on the GPU
  CHECK( cudaMalloc( (void**)&dev_M, N*N*sizeof(int)) );
  CHECK( cudaMalloc( (void**)&dev_T, N*N*sizeof(int)) );

  // Initialise the values of the matrix on the CPU
  int k = 0;
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      M[i][j] = k++;
    }
  }

  // Copy the input arrays from host to device
  CHECK( cudaMemcpy( dev_M, M, N*N*sizeof(int), cudaMemcpyHostToDevice) );
  CHECK( cudaMemcpy( dev_T, T, N*N*sizeof(int), cudaMemcpyHostToDevice) );

  dim3 grid(N, N);
  transpose<<<grid,1>>>( dev_M, dev_T );

  // Copy the results back from device to host array
  CHECK( cudaMemcpy( T, dev_T, N*N*sizeof(int), cudaMemcpyDeviceToHost) );

  // Output result
  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      printf( "%d  ", T[i][j]);
    }
    printf("\n");
  }

  // Clean up device.
  cudaFree( dev_M );
  cudaFree( dev_T );

  return 0;
}
