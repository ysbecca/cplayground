/*
@author ysbecca Rebecca Young [sc16rsmy]
@description A Sobel image filter on GPU

COMP5811 Parallel & Concurrent Programming
University of Leeds 2016-2017
Prof. David Duke
*/

#include <stdio.h>
#include "sobel_filter.h"
#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }
#define THRESH 10000

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
  grayScale.img = (unsigned char *)malloc(pixels);
  for (int i = 0; i < pixels; i++)
  {
      unsigned int r = source.img[i*3];
      unsigned int g = source.img[i*3 + 1];
      unsigned int b = source.img[i*3 + 2];
      grayScale.img[i] = 0.2989*r + 0.5870*g + 0.1140*b;
  }


  Image filtered;
  filtered.width = source.width;
  filtered.height = source.height;
  filtered.img = (unsigned char *)malloc(pixels);





  // Writes a filtered image back to my_sobel.pgm
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

  free(grayScale.img);
  free(source.img);
  free(filtered.img);

  exit(0);
}
