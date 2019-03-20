///////////////////////////////////////////////////
//          Call to general libraries            //
///////////////////////////////////////////////////
#include <cstdio>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda_gl_interop.h>
#include "UI.cu"
using namespace std;


///////////////////////////////////////////////////
//                  Main function                //
///////////////////////////////////////////////////
int main(int argc, char** argv){
  time_t t;
  srand((unsigned) time(&t));
  int i = 0, j;
  unsigned char c;
  Scale_Factor = 1;
  Rotation_Factor = 0;
  //////// Reading data from images
  char T[1024]="montaña.bmp";
  fp = fopen(T,"r");
  Read_Image(fp, &Image_Raw);
  i = 54;
  while(i < Image_Raw.offset) {
    c = fgetc(fp);
    if(feof(fp))
       break;
    i++;
  }
  ////////Building RGB matrix 
  Num_Cols = Image_Raw.widht, Num_Rows = Image_Raw.height;
  sizeImage = Num_Cols*Num_Rows;
  unsigned char RR[Num_Rows][Num_Cols], GG[Num_Rows][Num_Cols], BB[Num_Rows][Num_Cols];
  for ( i = 0; i < Num_Rows; i++) {
    for( j = 0; j < Num_Cols; j++){
      if (Image_Raw.bitsPerPixel > 8) {
        BB[i][j] = fgetc(fp);
        GG[i][j] = fgetc(fp);
        RR[i][j] = fgetc(fp);
        if(Image_Raw.bitsPerPixel > 24)
          c = fgetc(fp);
      } else if(Image_Raw.bitsPerPixel == 8) {
        c = getc(fp);
        BB[i][j] = c;
        GG[i][j] = c;
        RR[i][j] = c;}
    }
  }
  fclose(fp);

  widht = WIDTH; height = HEIGHT;

  Num_Rows_Fourier = pow(2,(int)(log(Num_Rows - 1)/log(2)) + 1);
  Num_Cols_Fourier = pow(2,(int)(log(Num_Cols - 1)/log(2)) + 1);
  unsigned int his_size = sizeof(unsigned int)*256;
  unsigned int comp_size = sizeof(float)*Num_Rows_Fourier*Num_Cols_Fourier;

  cudaMallocManaged(&d_his_r, his_size);
  cudaMallocManaged(&d_his_g, his_size);
  cudaMallocManaged(&d_his_b, his_size);

  cudaMallocManaged(&Val_Real, comp_size);
  cudaMallocManaged(&Val_Real_out, comp_size);
  cudaMallocManaged(&Val_Imag, comp_size);
  cudaMallocManaged(&Val_Imag_out, comp_size);

  cudaMallocManaged(&DMask, sizeof(float)*625);
///////////////////////////////////////////////////
//                  Cuda memory                 //
///////////////////////////////////////////////////
  // Original
  cudaMallocPitch((void**)&Image_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Image_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Image_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  // Back - up
  cudaMallocPitch((void**)&Image_R_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Image_G_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Image_B_bk, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  // Fourier
  cudaMallocPitch((void**)&Fourier_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Fourier_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Fourier_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Val_Real, &fou_size, sizeof(float)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Val_Imag, &fou_size, sizeof(float)*Num_Cols, Num_Rows);
  // Convolution
  cudaMallocPitch((void**)&Convol_R, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Convol_G, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Convol_B, &or_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  // Equalization
  cudaMallocPitch((void**)&Equalizar_R, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Equalizar_G, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  cudaMallocPitch((void**)&Equalizar_B, &equ_size, sizeof(unsigned char)*Num_Cols, Num_Rows);
  // Rotate and scaling
  cudaMallocPitch((void**)&Morfo_R, &mor_size, sizeof(unsigned char)*DIM, DIM);
  cudaMallocPitch((void**)&Morfo_G, &mor_size, sizeof(unsigned char)*DIM, DIM);
  cudaMallocPitch((void**)&Morfo_B, &mor_size, sizeof(unsigned char)*DIM, DIM);

  // Copying to GPU
  cudaMemcpy2D(Image_R, or_size, RR, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  cudaMemcpy2D(Image_G, or_size, GG, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  cudaMemcpy2D(Image_B, or_size, BB, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  cudaMemcpy2D(Image_R_bk, or_size, RR, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  cudaMemcpy2D(Image_G_bk, or_size, GG, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  cudaMemcpy2D(Image_B_bk, or_size, BB, sizeof(unsigned char)*Num_Cols,
               sizeof(unsigned char)*Num_Cols, Num_Rows, cudaMemcpyHostToDevice);

  glutInitWindowSize(widht, height);
  glutInit(&argc, argv);
  glutInitContextFlags(GLUT_DEBUG);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL | GLUT_DOUBLE);
  glutCreateWindow("Editor de Imagenes");

  glewInit();

  if (GLEW_KHR_debug){
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  } else {
    printf("No GLEW_KHR_debug!");}

  Create_call_back_function();
  glutDisplayFunc(display);
  glutMainLoop();

  cudaFree(Image_R), cudaFree(Image_G), cudaFree(Image_B);
  cudaFree(Morfo_R), cudaFree(Morfo_G), cudaFree(Morfo_B);
  cudaFree(Fourier_R), cudaFree(Fourier_G), cudaFree(Fourier_B);
  cudaFree(Convol_R), cudaFree(Convol_G), cudaFree(Convol_B);
  cudaFree(Val_Real), cudaFree(Val_Real);
  return 0;

}
