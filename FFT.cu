#include <curand.h>
#include <curand_kernel.h>

#define DIM 1600
#define PI 3.14159265

__device__ int log2(int N){
  int k = N, i = 0;
  while(k) {
    k >>= 1;
    i++;}
  return i - 1;
}

__device__ int reverse(int N, int n) {
  int p = 0;
  for(int j = 1; j <= log2(N); j++) {
    if(n & (1 << (log2(N) - j)))
      p |= 1 << (j - 1);
  }
  return p;
}


__device__ void ordina_x(float *complex_r, float *complex_i,
                      float *real_d_out, float *imagi_d_out,
                      int row, int col, int x) {
  int N = row, a;
  for(int i = 0; i < N; i++){
    a = reverse((int)N, i);
    real_d_out[i*col + x] = complex_r[a*col + x];
    imagi_d_out[i*col + x] = complex_i[a*col + x];}
  for(int j = 0; j < N; j++){
    complex_r[j*col + x] = real_d_out[j*col + x];
    complex_i[j*col + x] = imagi_d_out[j*col + x];}
}

__device__ void ordina_y(float *complex_r, float *complex_i,
                      float *real_d_out, float *imagi_d_out,
                      int row, int col, int y) {
  int N = row, a;
  for(int i = 0; i < N; i++){
    a = reverse((int)N, i);
    real_d_out[y*col + i] = complex_r[y*col + a];
    imagi_d_out[y*col + i] = complex_i[y*col + a];}
  for(int j = 0; j < N; j++){
    complex_r[y*col + j] = real_d_out[y*col + j];
    complex_i[y*col + j] = imagi_d_out[y*col + j];}
}

__device__ void Func_FFT_X(float *complex_r, float *complex_i,
                     int row, int col, int x){
  int n = 1, N = row;
  int a = N/2;
  float temp_real, temp_imagi;
  float t_r, t_i, a_r, a_i;
  for(int j = 0; j < log2(N); j++){
    for (int i = 0; i < N; i++) {
      if(!(i & n)) {
        temp_real = complex_r[x + (i * col)];
        temp_imagi = complex_i[x + (i * col)];
        a_r = cos((-2) * ((i * a) % (n * a)) * PI / N);
        a_i = sin((-2) * ((i * a) % (n * a)) * PI / N);
        t_r = (a_r*complex_r[x + (i + n)*col]) - (a_i*complex_i[x + (i + n)*col]);
        t_i = (a_i*complex_r[x + (i + n)*col]) + (a_r*complex_i[x + (i + n)*col]);
        complex_r[x + (i * col)] += t_r;
        complex_i[x + (i * col)] += t_i;
        complex_r[x + (i + n)*col] = temp_real - t_r;
        complex_i[x + (i + n)*col] = temp_imagi - t_i;}
    }
    n *= 2;
    a = a/2;
  }
}

__device__ void Func_FFT_Y(float *complex_r, float *complex_i,
                     int row, int col, int y){
  int n = 1, N = col;
  int a = N/2;
  float temp_real, temp_imagi;
  float t_r, t_i, a_r, a_i;
  for(int j = 0; j < log2(N); j++){
    for (int i = 0; i < N; i++) {
      if(!(i & n)) {
        temp_real = complex_r[i + (y * col)];
        temp_imagi = complex_i[i + (y * col)];
        a_r = cos(-2 * ((i * a) % (n * a)) * PI/ N);
        a_i = sin(-2 * ((i * a) % (n * a)) * PI/ N);
        t_r = (a_r*complex_r[(i + n) + y*col]) - (a_i*complex_i[(i + n) + y*col]);
        t_i = (a_i*complex_r[(i + n) + y*col]) + (a_r*complex_i[(i + n) + y*col]);
        complex_r[i + (y * col)] += t_r;
        complex_i[i + (y * col)] += t_i;
        complex_r[(i + n) + y*col] = temp_real - t_r;
        complex_i[(i + n) + y*col] = temp_imagi - t_i;}
    }
    n *= 2;
    a = a/2;
  }
}

__global__ void FFT_X(unsigned char *R_input, unsigned char *G_input,
                    unsigned char *B_input, size_t i_size,
                    float *complex_r, float *complex_i,
                    float *real_d_out, float *imagi_d_out,
                    unsigned char *r_dataC, unsigned char *g_dataC,
                    unsigned char *b_dataC, unsigned long col, unsigned long row,
                    unsigned long colF, unsigned long rowF ) {
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  float temp;
  if(x < col){
    for (int i = 0; i < row; i++) {
      complex_r[x + (i * colF)] = 0.2989 * R_input[x + (i * i_size)] +  0.587 * G_input[x + (i * i_size)] + 0.1140 * B_input[x + (i * i_size)];
      complex_i[x + (i * colF)] = 0;}
    for (int i = row; i < rowF; i++) {
      complex_r[x + (i * colF)] = 0;
      complex_i[x + (i * colF)] = 0;}
  }else{
    for (int i = 0; i < rowF; i++) {
      complex_r[x + (i * colF)] = 0;
      complex_i[x + (i * colF)] = 0;}
  }
  ordina_x(complex_r, complex_i, real_d_out, imagi_d_out, rowF, colF, x);
  Func_FFT_X(complex_r, complex_i, rowF, colF, x);
  for (int i = 0; i < rowF/2; i++){
    temp = complex_r[x + (i * colF)];
    complex_r[x + (i * colF)] = complex_r[x + ((i + rowF/2) * colF)];
    complex_r[x + ((i + rowF/2) * colF)] = temp;
    temp = complex_i[x + (i * colF)];
    complex_i[x + (i * colF)] = complex_i[x + ((i + rowF/2) * colF)];
    complex_i[x + ((i + rowF/2) * colF)] = temp;}
}

__global__ void FFT_Y(unsigned char *R_input, unsigned char *G_input,
                    unsigned char *B_input, size_t i_size,
                    float *complex_r, float *complex_i,
                    float *real_d_out, float *imagi_d_out,
                    unsigned char *r_dataC, unsigned char *g_dataC,
                    unsigned char *b_dataC, unsigned long col, unsigned long row,
                    unsigned long colF, unsigned long rowF ) {
  int y = threadIdx.x + (blockIdx.x * blockDim.x);
  float temp;
  ordina_y(complex_r, complex_i, real_d_out, imagi_d_out, rowF, colF, y);
  Func_FFT_Y(complex_r, complex_i, rowF, colF, y);
  for (int i = 0; i < colF/2; i++) {
    temp = complex_r[i + (y * colF)];
    complex_r[i + (y * colF)] = complex_r[(i + colF/2) + (y * colF)];
    complex_r[(i + colF/2) + (y * colF)] = temp;
    temp = complex_i[i + (y * colF)];
    complex_i[i + (y * colF)] = complex_i[(i + colF/2) + (y * colF)];
    complex_i[(i + colF/2) + (y * colF)] = temp;}

  unsigned char v;
  int a = (colF/2) - (col/2);
  int temp_b = (rowF/2) - (row/2);
  if( y >= temp_b)
    for (int i = a; i < (colF/2) + (col/2); i++) {
      v = (unsigned char)(20*log10(sqrt((complex_r[i + (y * colF)]*complex_r[i + (y * colF)]) + (complex_i[i + (y * colF)]*complex_i[i + (y * colF)]))));
      r_dataC[(i - a ) + (y - temp_b) * i_size] = v;
      g_dataC[(i - a) + (y - temp_b) * i_size] = v;
      b_dataC[(i - a) + (y - temp_b) * i_size] = v;}
}



