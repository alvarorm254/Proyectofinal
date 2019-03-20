#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void Get_Histogram(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned int *hist_r,unsigned int *hist_g,unsigned int *hist_b) {

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;

  atomicAdd( &(hist_r[R_input[offset]]), 1);
  atomicAdd( &(hist_g[G_input[offset]]), 1);
  atomicAdd( &(hist_b[B_input[offset]]), 1);
}

__global__ void Equalization_GPU(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned char *r_dataE, unsigned char *g_dataE,
                        unsigned char *b_dataE,
                        unsigned int *hist_r,unsigned int *hist_g,unsigned int *hist_b) {

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  r_dataE[offset] = hist_r[R_input[offset]];
  g_dataE[offset] = hist_g[G_input[offset]];
  b_dataE[offset] = hist_b[B_input[offset]];
}

