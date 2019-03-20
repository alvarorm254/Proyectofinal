#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void Rotate(uchar4 *ptr, unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size, float a,
                        unsigned long col, unsigned long row)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * blockDim.x * gridDim.x;
  x = x - (blockDim.x * gridDim.x / 2);
  y = y - (blockDim.y * gridDim.y / 2);

  unsigned char* f_r, *f_g, *f_b;

  int ximg = (x*cos(a) + y*sin(a)) + (col/2), yimg = (y*cos(a) - x*sin(a)) + (row/2);
  if (ximg < col && yimg < row) {
    f_r = (unsigned char*)((char*)R_input + yimg*i_size);
    f_g = (unsigned char*)((char*)G_input + yimg*i_size);
    f_b = (unsigned char*)((char*)B_input + yimg*i_size);
    ptr[offset].x = f_r[ximg];
    ptr[offset].y = f_g[ximg];
    ptr[offset].z = f_b[ximg];
    ptr[offset].w = 255;
  } else{
    ptr[offset].x = 0;
    ptr[offset].y = 0;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
  }

}

__global__ void Scale(unsigned char *R_input, unsigned char *G_input,unsigned char *B_input,
                        unsigned char *R_output, unsigned char *G_output,unsigned char *B_output,
                        size_t i_size, size_t pitch2, float s,
                        unsigned long col, unsigned long row){
  float x = threadIdx.x + (blockIdx.x * blockDim.x);
  float y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * pitch2;
  x = x - (DIM / 2);
  y = y - (DIM / 2);

  unsigned char* f_r, *f_g, *f_b;
  x /= s; y /= s;

  int ximg = x + (col/2), yimg = y + (row/2);
  if (ximg < (col - 1) && yimg < (row - 1)) {
    f_r = (unsigned char*)((char*)R_input + yimg*i_size);
    f_g = (unsigned char*)((char*)G_input + yimg*i_size);
    f_b = (unsigned char*)((char*)B_input + yimg*i_size);
    float cx = x - floor(x);
    float cy = y - floor(y);
    float R1 = f_r[ximg]*(1 - cx) + f_r[ximg + 1]*(cx);
    float R2 = f_r[ximg + i_size]*(1 - cx) + f_r[ximg + 1 + i_size]*(cx);
    R_output[offset] = R1*(1 - cy) + R2*(cy);

    R1 = f_g[ximg]*(1 - cx) + f_g[ximg + 1]*(cx);
    R2 = f_g[ximg + i_size]*(1 - cx) + f_g[ximg + 1 + i_size]*(cx);
    G_output[offset] = R1*(1 - cy) + R2*(cy);

    R1 = f_b[ximg]*(1 - cx) + f_b[ximg + 1]*(cx);
    R2 = f_b[ximg + i_size]*(1 - cx) + f_b[ximg + 1 + i_size]*(cx);
    B_output[offset] = R1*(1 - cy) + R2*(cy);
  }else{
    R_output[offset] = 0;
    G_output[offset] = 0;
    B_output[offset] = 0;
  }
}
