/*
@author ysbecca Rebecca Stone (Young) [sc16rsmy]
@description Sobel filter implementation
*/

#ifndef ADD_SOBEL_TEXTURES
#define ADD_SOBEL_TEXTURES

__global__
void apply_sobel_textures(unsigned char *output) {

  // Pixel coordinates for this thread
  int x = blockIdx.x;
  int y = blockIdx.y;
  int tid = (gridDim.x * y) + x;
  // Top, top left, top right, left, right, bottom left, bottom right, bottom
  int top = (gridDim.x * (y-1)) + x;
  int top_left = (gridDim.x * (y-1)) + (x-1);
  int top_right = (gridDim.x * (y-1)) + (x+1);
  int left = (gridDim.x * y) + (x-1);
  int right = (gridDim.x * y) + (x+1);
  int bottom = (gridDim.x * (y+1)) + x;
  int bottom_left = (gridDim.x * (y+1)) + (x-1);
  int bottom_right = (gridDim.x * (y+1)) + (x+1);

  // Ignore boundaries
  if(x < WIDTH-1 && x > 0 && y < HEIGHT-1 && y > 0) {
    int grad_x = tex1Dfetch(tex_input, top_right) - tex1Dfetch(tex_input, top_left) + 2*tex1Dfetch(tex_input, right) - 2*tex1Dfetch(tex_input, left) + tex1Dfetch(tex_input, bottom_right) - tex1Dfetch(tex_input, bottom_left);
    int grad_y = tex1Dfetch(tex_input, top_left) + 2*tex1Dfetch(tex_input, top) + tex1Dfetch(tex_input, top_right) - tex1Dfetch(tex_input, bottom_left) - 2*tex1Dfetch(tex_input, bottom) - tex1Dfetch(tex_input, bottom_right);
    int magnitude = (grad_x * grad_x) + (grad_y * grad_y);
    if(magnitude > THRESHOLD) {
      output[tid] = 255;
    } else {
      output[tid] = 0;
    }
  } else {
    output[tid] = 0;
  }

}







#endif
