#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__global__ void erode(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned char *r_dataC, unsigned char *g_dataC,
                        unsigned char *b_dataC, unsigned long col, unsigned long row,
                        unsigned int dim, int m) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  int offset2, ximg, yimg;
  int c1 = 255,c2 = 255,c3 = 255;
  int end = dim/2, ini = -end;

  for (int i = ini; i <= end; i++) {
    ximg = x + i;
    for (int j = ini; j <= end; j++) {
      yimg = y + j;
      offset2 = ximg + yimg * i_size;
      if (ximg < col && yimg < row)
        if (ximg > 0 && yimg > 0)
          if(R_input[offset2]+G_input[offset2]+B_input[offset2]<c1+c2+c3)
          c1 = R_input[offset2];
          c2 = G_input[offset2];
          c3 = B_input[offset2];
    }
  }
  r_dataC[offset] = c1;
  g_dataC[offset] = c2;
  b_dataC[offset] = c3;
}

__global__ void dilate(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned char *r_dataC, unsigned char *g_dataC,
                        unsigned char *b_dataC, unsigned long col, unsigned long row,
                        unsigned int dim, int m) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  int offset2, ximg, yimg;
  int c1 = 0,c2 = 0,c3 = 0;
  int end = dim/2, ini = -end;

  for (int i = ini; i <= end; i++) {
    ximg = x + i;
    for (int j = ini; j <= end; j++) {
      yimg = y + j;
      offset2 = ximg + yimg * i_size;
      if (ximg < col && yimg < row)
        if (ximg > 0 && yimg > 0)
          if(R_input[offset2]+G_input[offset2]+B_input[offset2]>c1+c2+c3)
          c1 = R_input[offset2];
          c2 = G_input[offset2];
          c3 = B_input[offset2];
    }
  }
  r_dataC[offset] = c1;
  g_dataC[offset] = c2;
  b_dataC[offset] = c3;
}

__global__ void median_filter(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned char *r_dataC, unsigned char *g_dataC,
                        unsigned char *b_dataC, unsigned long col, unsigned long row,
                        unsigned int dim) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  int offset2, ximg, yimg;
  unsigned char temp_r = 0, temp_g = 0, temp_b = 0, temp;
  int end = dim/2, ini = -end, k = 0, n = 0, i, j;
  int hr[9];
  int hg[9];
  int hb[9];

  for (i = ini; i <= end; i++) {
    ximg = x + i;
    for (j = ini; j <= end; j++) {
      yimg = y + j;
      offset2 = ximg + yimg * i_size;
      if (ximg < col && yimg < row)
        if (ximg > 0 && yimg > 0) {
          hr[n] = R_input[offset2];
          hg[n] = G_input[offset2];
          hb[n] = B_input[offset2];
          n++;}
      k++;
    }
  }
  for (i = 0; i < n; i++)
    for (j= i + 1; j < n; j++)
      if (hr[j] < hr[i]) {
        temp = hr[j];
        hr[j] = hr[i];
        hr[i] = temp;}

  for (i = 0; i < n; i++)
    for (j= i + 1; j < n; j++)
      if (hg[j] < hg[i]) {
        temp = hg[j];
        hg[j] = hg[i];
        hg[i] = temp;}

  for (i = 0; i < n; i++)
    for (j= i + 1; j < n; j++)
      if (hb[j] < hb[i]) {
        temp = hb[j];
        hb[j] = hb[i];
        hb[i] = temp;}

  if(n%2 == 1){
    temp_r = hr[(n/2)];
    temp_g = hg[(n/2)];
    temp_b = hb[(n/2)];
  }else{
    temp_r = hr[(n/2)] + hr[(n/2) - 1];
    temp_g = hg[(n/2)] + hg[(n/2) - 1];
    temp_b = hb[(n/2)] + hb[(n/2) - 1];}

  r_dataC[offset] = temp_r;
  g_dataC[offset] = temp_g;
  b_dataC[offset] = temp_b;
}
__global__ void Operador_Convolucion(unsigned char *R_input, unsigned char *G_input,
                        unsigned char *B_input, size_t i_size,
                        unsigned char *r_dataC, unsigned char *g_dataC,
                        unsigned char *b_dataC, unsigned long col, unsigned long row,
                        float *mask, unsigned int dim) {

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int offset = x + y * i_size;
  int offset2, ximg, yimg;
  unsigned char temp_r = 0, temp_g = 0, temp_b = 0;
  int end = dim/2, ini = -end, k = 0;

  for (int i = ini; i <= end; i++) {
    ximg = x + i;
    for (int j = ini; j <= end; j++) {
      yimg = y + j;
      offset2 = ximg + yimg * i_size;
      if (ximg < col && yimg < row)
        if (ximg > 0 && yimg > 0) {
          temp_r += R_input[offset2]*mask[k];
          temp_g += G_input[offset2]*mask[k];
          temp_b += B_input[offset2]*mask[k];}
      k++;
    }
  }
  r_dataC[offset] = temp_r;
  g_dataC[offset] = temp_g;
  b_dataC[offset] = temp_b;
}

