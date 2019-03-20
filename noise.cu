#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void PPnoise(unsigned char *R_input, unsigned char *G_input,
                      unsigned char *B_input, size_t i_size, int noiseP, int seed){
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  curandState_t state;
  curand_init(seed, x,  y, &state);

  unsigned char noise = (unsigned char)(curand(&state) % 100);
  if(curand(&state) % 100 < noiseP){
    noise = 255 * (noise % 2);
    R_input[offset] = noise;
    G_input[offset] = noise;
    B_input[offset] = noise;
  }
}
