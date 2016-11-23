
#include <stdio.h>
#include "check.h"
#define N 8 // Matrix dimension
#define TILES 4 // How many tiles along one coordinate

__global__
void transpose(int *M, int *T)
{
  //__shared__ int cache[N/TILES*N/TILES]; // 8x8

  // Global position
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tid1 = ((gridDim.x * blockDim.x) * idy) + idx; // Value we want to copy
  // Transposed within tile position
  int idx2 = (blockIdx.x * blockDim.x) + threadIdx.y;
  int idy2 = (blockIdx.y * blockDim.y) + threadIdx.x;
  int tid2 = ((gridDim.x * blockDim.x) * idy2) + idx2;

  if (tid1 < N*N) {
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

  dim3 grid(N/TILES, N/TILES);
  int threads_dim = N / TILES; // Threads per tile across one coordinate.
  dim3 threads(TILES, TILES);

  printf( "Matrix of (%d by %d) with (%d by %d) thread blocks.\n", N, N, threads_dim, threads_dim);

  transpose<<<grid,threads>>>( dev_M, dev_T );

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
