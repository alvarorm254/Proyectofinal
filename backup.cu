#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void Backup(unsigned char *R_input, unsigned char *G_input,
                      unsigned char *B_input, size_t i_size,
                      unsigned char *R_output, unsigned char *G_output,
                      unsigned char *B_output){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  R_output[offset] = R_input[offset];
  G_output[offset] = G_input[offset];
  B_output[offset] = B_input[offset];
}



