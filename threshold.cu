#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265


__global__ void grayscale(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned int *hist) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  R_input[offset] = 0.2989 * R_input[offset] +  0.587 * G_input[offset] + 0.1140 * B_input[offset];
  G_input[offset] = 0.2989 * R_input[offset] +  0.587 * G_input[offset] + 0.1140 * B_input[offset];
  B_input[offset] = 0.2989 * R_input[offset] +  0.587 * G_input[offset] + 0.1140 * B_input[offset];
  atomicAdd( &(hist[R_input[offset]]), 1);
}
__global__ void binary(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        int um) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  unsigned char c;
  if (R_input[offset] > um) c = 255;
  else c = 0;
  R_input[offset] = c;
  G_input[offset] = c;
  B_input[offset] = c;
}
