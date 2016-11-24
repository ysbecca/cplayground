/*
@author ysbecca Rebecca Young [sc16rsmy]
@description A Sobel image filter on GPU

COMP5811 Parallel & Concurrent Programming
University of Leeds 2016-2017
Prof. David Duke
*/

#include <stdio.h>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }
#define THRESHOLD 10000
#define WIDTH 640
#define HEIGHT 480

#include "sobel_filter.h"

// Image data structure.
struct Image {
  int width;
  int height;
  unsigned char *img;
  unsigned char *dev_img;
};

int main(int argc, char **argv)
{
  Image source;

  if (argc != 2)
  {
      printf("Usage: exec filename\n");
      exit(1);
  }
  char *fname = argv[1];
  FILE *src;

  if (!(src = fopen(fname, "rb")))
  {
      printf("Couldn't open file %s for reading.\n", fname);
      exit(1);
  }

  char p,s;
  fscanf(src, "%c%c\n", &p, &s);
  if (p != 'P' || s != '6')
  {
      printf("Not a valid PPM file (%c %c)\n", p, s);
      exit(1);
  }

  fscanf(src, "%d %d\n", &source.width, &source.height);
  int ignored;
  fscanf(src, "%d\n", &ignored);

  int pixels = source.width * source.height;
  source.img = (unsigned char *)malloc(pixels*3);
  if (fread(source.img, sizeof(unsigned char), pixels*3, src) != pixels*3)
    {
       printf("Error reading file.\n");
       exit(1);
    }
  fclose(src);

  Image grayScale;
  grayScale.width = source.width;
  grayScale.height = source.height;
  printf("Width %d, height %d\n", source.width, source.height);
  grayScale.img = (unsigned char *)malloc(pixels);
  for (int i = 0; i < pixels; i++)
  {
      unsigned int r = source.img[i*3];
      unsigned int g = source.img[i*3 + 1];
      unsigned int b = source.img[i*3 + 2];
      grayScale.img[i] = 0.2989*r + 0.5870*g + 0.1140*b;
  }
  // The structure on CPU for the filtered image to be saved into.
  Image filtered;
  filtered.width = source.width;
  filtered.height = source.height;
  filtered.img = (unsigned char *)malloc(pixels);

  // Allocate memory for the images on the GPU
  CHECK( cudaMalloc( (void**)&grayScale.dev_img, pixels ) );
  CHECK( cudaMalloc( (void**)&filtered.dev_img, pixels ) );

  // Copy the input arrays from host to device
  CHECK( cudaMemcpy( grayScale.dev_img, grayScale.img, pixels, cudaMemcpyHostToDevice) );
  CHECK( cudaMemcpy( filtered.dev_img, filtered.img, pixels, cudaMemcpyHostToDevice) );

  // One thread per pixel; assume image size /32
  dim3 grid(source.width, source.height);

  // Start the timer.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Do work on GPU.
  apply_sobel<<<grid,1>>>( grayScale.dev_img, filtered.dev_img );

  // Stop time.
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Copy the results back from device to host image
  CHECK( cudaMemcpy( filtered.img, filtered.dev_img, pixels, cudaMemcpyDeviceToHost) );

  // Display the elapsed time
  float t;
  cudaEventElapsedTime(&t, start, stop);
  printf("GPU took %f to complete task.\n", t);

  // Writes the filtered image back to my_sobel.pgm
  FILE *out;
  if (!(out = fopen("my_sobel.pgm", "wb")))
  {
      printf("Couldn't open file for output.\n");
      exit(1);
  }
  fprintf(out, "P5\n%d %d\n255\n", filtered.width, filtered.height);
  if (fwrite(filtered.img, sizeof(unsigned char), pixels, out) != pixels)
  {
      printf("Error writing file.\n");
      exit(1);
  }
  fclose(out);
  // Tidy-up everything
  free(grayScale.img);
  free(source.img);
  free(filtered.img);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  exit(0);
}
