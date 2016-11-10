#include <stdio.h>
#include "check.h"

int main(int argc, char**argv)
{
  cudaDeviceProp prop;

  int count;
  CHECK( cudaGetDeviceCount(&count) );
  for (int i = 0; i < count; i++)
  {
      CHECK( cudaGetDeviceProperties( &prop, i) );
      printf("Device %d --------------------\n", i);
      printf("Name: %s\n", prop.name);
      printf("Compute capability: %d.%d\n", prop.major, prop.minor);
      printf("Clock rate: %d\n", prop.clockRate);
      printf("Device copy overlap: %s\n", prop.deviceOverlap ? "enabled" : "disabled");
      printf("Kernel exec timeout: %s\n", prop.kernelExecTimeoutEnabled ? "enabled" : "disabled");
      printf("Total global mem: %ld\n", prop.totalGlobalMem);
      printf("Total const mem: %ld\n", prop.totalConstMem);
      printf("Max mem pitch: %ld\n", prop.memPitch);
      printf("Texture alignment: %ld\n", prop.textureAlignment);
      printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
      printf("Shared mem per MP: %ld\n", prop.sharedMemPerBlock);
      printf("Registers per MP: %d\n" , prop.regsPerBlock);
      printf("Threads in warp: %d\n", prop.warpSize);
      printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
      printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
      printf("\n");
  }
}

