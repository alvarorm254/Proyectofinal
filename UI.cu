//nvcc Final.cu -o out -lglut -lGLEW -lGL -lm -ccbin clang-3.8 -lstdc++
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
///////////////////////////////////////////////////
//            Call to cuda funtions              //
///////////////////////////////////////////////////
#include "FFT.cu"
#include "convolutions.cu"
#include "threshold.cu"
#include "RE_transform.cu"
#include "equalization.cu"
#include "noise.cu"
#include "backup.cu"
#include "compression.cu"


using namespace std;

///////////////////////////////////////////////////
//              Defining variables               //
///////////////////////////////////////////////////
#define WIDTH  1280
#define HEIGHT 960
#define DIM 1600
#define PI 3.14159265

static int sub_00;
static int sub_01;
static int sub_02;
static int sub_03;
static int sub_04;

bool Equalization = 0, Flag_Filt = 0, Flag_Med = 0, Flag_PPnoise = 0,Flag_Pix = 0,Flag_req=0;
bool Flag_Reset = 0, Flag_Ero = 0, Flag_Dil = 0, Flag_Gray = 0, Flag_BW = 0, Flag_Fourier = 0;
long long int sizeImage;
float Scale_Factor;
float Rotation_Factor;
unsigned long widht, height;
int Num_Cols, Num_Rows, Dim_Con, Num_Rows_Fourier, Num_Cols_Fourier, Max_E;
size_t or_size, mor_size, equ_size, fou_size;
unsigned char *Image_R, *Image_G, *Image_B;
unsigned char *Image_R_bk, *Image_G_bk, *Image_B_bk;
unsigned char *Equalizar_R, *Equalizar_G, *Equalizar_B;
unsigned char *Convol_R, *Convol_G, *Convol_B;
unsigned char *Fourier_R, *Fourier_G, *Fourier_B;
unsigned char *Morfo_R, *Morfo_G, *Morfo_B;
float *Val_Real, *Val_Real_out, *Val_Imag, *Val_Imag_out;
unsigned int *d_his_r;
unsigned int *d_his_g;
unsigned int *d_his_b;
float *DMask;
float *Mask = (float*)malloc(625*sizeof(float));

///////////////////////////////////////////////////
//          Serial part of cuda funtions         //
///////////////////////////////////////////////////

int Threshold(unsigned char *r_data, unsigned char *g_data, unsigned char *b_data, size_t pitch);

void Equalization_PC (unsigned char *r_data, unsigned char *g_data,
                     unsigned char *b_data, size_t pitch,
                     unsigned char *r_dataE, unsigned char *g_dataE,
                     unsigned char *b_dataE );

void FFT();

///////////////////////////////////////////////////
//         Function to display with glut         //
///////////////////////////////////////////////////

void display(){
  GLuint bufferObj;
  struct cudaGraphicsResource* resource;
  bool Flag_conv = 1;
  glClearColor( 255.0, 255.0, 255.0, 1.0  );
  glClear( GL_COLOR_BUFFER_BIT );

  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, widht * height * 4, NULL, GL_DYNAMIC_DRAW_ARB );
  cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );
  uchar4* devPtr;
  size_t size;
  cudaGraphicsMapResources( 1, &resource, NULL ) ;
  cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, resource );

  dim3 grids(Num_Cols/16,Num_Rows/16);
  dim3 threads(16, 16);
  dim3 grids_01(DIM/16,DIM/16);
  dim3 threads_01(16, 16);
  dim3 grids_02(widht/16,height/16);
  dim3 threads_02(16, 16);
///////////////////////////////////////////////////
//       Cuda functions called by the menu       //
///////////////////////////////////////////////////
  if(Flag_Pix){
    Pixelado<<<grids,threads>>>(Image_R, Image_G, Image_B,or_size, Image_R, Image_G, Image_B);
    Flag_Pix = 0;}
  if(Flag_req){
    Requant<<<grids,threads>>>(Image_R, Image_G, Image_B,or_size, Image_R, Image_G, Image_B);
    Flag_req = 0;}

  if(Flag_Reset){
    Backup<<<grids,threads>>>(Image_R_bk, Image_G_bk, Image_B_bk, or_size,
      Image_R, Image_G, Image_B);
    Flag_Reset = 0;}

  if(Flag_Ero){
    erode<<<grids,threads>>>(Image_R, Image_G, Image_B,
      or_size, Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Dim_Con, Max_E);
    Flag_Ero = 0;}
  if(Flag_Dil){
    dilate<<<grids,threads>>>(Image_R, Image_G, Image_B,
      or_size, Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Dim_Con, Max_E);
    Flag_Dil = 0;}
  if(Flag_Gray){
    grayscale<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size, d_his_r);
    Flag_Gray = 0;}

  if(Flag_BW){
    Threshold (Image_R, Image_G, Image_B, or_size );
    Flag_BW = 0;}

  if (Flag_PPnoise){
    PPnoise<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size, 1, rand()%100);
    Flag_PPnoise = 0;}

  if(Flag_Fourier){
    FFT();
    Flag_Fourier = 0;}

  if (Equalization){
    Equalization_PC (Image_R, Image_G, Image_B, or_size,Convol_R, Convol_G, Convol_B );
    Backup<<<grids,threads>>>(Convol_R, Convol_G, Convol_B, or_size,
      Image_R, Image_G, Image_B); 
    Equalization=0;   
  }
  if (Flag_Med) {
    median_filter<<<grids,threads>>>(Image_R, Image_G, Image_B, or_size,
      Convol_R, Convol_G, Convol_B, Num_Cols, Num_Rows, 3);
    Backup<<<grids,threads>>>(Convol_R, Convol_G, Convol_B, or_size,
      Image_R, Image_G, Image_B);
    Flag_Med=0;
  }
  if (Flag_Filt) {
    Operador_Convolucion<<<grids,threads>>>(Image_R, Image_G, Image_B,
      or_size, Convol_R, Convol_G, Convol_B, Num_Cols, Num_Rows, DMask, Dim_Con);
    Backup<<<grids,threads>>>(Convol_R, Convol_G, Convol_B, or_size,
      Image_R, Image_G, Image_B);
    Flag_Filt=0;
  }

  if (Flag_conv) {
    Scale<<<grids_01,threads_01>>>(Image_R, Image_G, Image_B, Morfo_R, Morfo_G, Morfo_B,
      or_size, mor_size, Scale_Factor, Num_Cols, Num_Rows);
  }else{
    Scale<<<grids_01,threads_01>>>(Convol_R, Convol_G, Convol_B, Morfo_R, Morfo_G, Morfo_B,
      or_size, mor_size, Scale_Factor, Num_Cols, Num_Rows);
  }


  Rotate<<<grids_02,threads_02>>>( devPtr, Morfo_R, Morfo_G, Morfo_B,
                              mor_size, Rotation_Factor, DIM, DIM);

  cudaGraphicsUnmapResources( 1, &resource, NULL ) ;
  glDrawPixels( widht, height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  glutSwapBuffers();
  cudaGraphicsUnregisterResource( resource ) ;
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
  glDeleteBuffers( 1, &bufferObj );
}

///////////////////////////////////////////////////
//                Serial funtions                //
///////////////////////////////////////////////////
int Threshold(unsigned char *r_data, unsigned char *g_data, unsigned char *b_data, size_t pitch) {
  unsigned int his_size = sizeof(unsigned int)*256;
  unsigned int *his = (unsigned int*)malloc(his_size);

  cudaMemset( d_his_r, 0, his_size);
  dim3 grids(Num_Cols,Num_Rows);
  dim3 threads(1, 1);
  grayscale<<<grids,threads>>>(r_data, g_data, b_data, pitch, d_his_r);
  cudaMemcpy(his, d_his_r, his_size, cudaMemcpyDeviceToHost);
  int m = Num_Cols*Num_Rows/2, h = 0, um, i;
  for (i = 0; i < 256; i++) {
    h += his[i];
    if (h > m) {
      um = i;
      break;
    }
  }
  binary<<<grids,threads>>>(r_data, g_data, b_data, pitch, um);
  return um;
}

void FFT(){
  FFT_X<<<Num_Cols_Fourier/128, 128>>>(Image_R, Image_G, Image_B,
                       or_size, Val_Real, Val_Imag, Val_Real_out, Val_Imag_out,
                       Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Num_Cols_Fourier, Num_Rows_Fourier);

  FFT_Y<<<Num_Rows_Fourier/128, 128>>>(Image_R, Image_G, Image_B,
                       or_size, Val_Real, Val_Imag, Val_Real_out, Val_Imag_out,
                       Image_R, Image_G, Image_B, Num_Cols, Num_Rows, Num_Cols_Fourier, Num_Rows_Fourier);
}

void Equalization_PC (unsigned char *r_data, unsigned char *g_data,
                     unsigned char *b_data, size_t pitch,
                     unsigned char *r_dataE, unsigned char *g_dataE,
                     unsigned char *b_dataE ){
  int i;
  unsigned int his_size = sizeof(unsigned int)*256;
  float hisAc_size = sizeof(float)*256;

  unsigned int *his_r = (unsigned int*)malloc(his_size);
  unsigned int *his_g = (unsigned int*)malloc(his_size);
  unsigned int *his_b = (unsigned int*)malloc(his_size);

  float *hisAc_r = (float*)malloc(hisAc_size);
  float *hisAc_g = (float*)malloc(hisAc_size);
  float *hisAc_b = (float*)malloc(hisAc_size);

  cudaMemset( d_his_r, 0, his_size);
  cudaMemset( d_his_g, 0, his_size);
  cudaMemset( d_his_b, 0, his_size);

  dim3 grids(Num_Cols,Num_Rows);
  dim3 threads(1, 1);
  Get_Histogram<<<grids,threads>>>(r_data, g_data, b_data, pitch, d_his_r, d_his_g, d_his_b);

  cudaMemcpy(his_r, d_his_r, his_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(his_g, d_his_g, his_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(his_b, d_his_b, his_size, cudaMemcpyDeviceToHost);
  float szImage=Num_Cols*Num_Rows;
  hisAc_r[0] = ((float)his_r[0]);
  hisAc_g[0] = ((float)his_g[0]);
  hisAc_b[0] = ((float)his_b[0]);
  for (i = 1; i < 256; i++) {
    hisAc_r[i] = hisAc_r[i-1] + (((float)his_r[i]));
    hisAc_g[i] = hisAc_g[i-1] + (((float)his_g[i]));
    hisAc_b[i] = hisAc_b[i-1] + (((float)his_b[i]));
  }
  his_r[0] = 0;
  his_g[0] = 0;
  his_b[0] = 0;

  for (i = 1; i < 255; i++) {
    his_r[i] = (int)(hisAc_r[i - 1]*255/szImage);
    his_g[i] = (int)(hisAc_g[i - 1]*255/szImage);
    his_b[i] = (int)(hisAc_b[i - 1]*255/szImage);
  }
  his_r[255] = 255;
  his_g[255] = 255;
  his_b[255] = 255;

  cudaMemcpy(d_his_r, his_r, his_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his_g, his_g, his_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his_b, his_b, his_size, cudaMemcpyHostToDevice);

  Equalization_GPU<<<grids,threads>>>(r_data, g_data, b_data,
    or_size, r_dataE, g_dataE, b_dataE, d_his_r, d_his_g, d_his_b);
}

///////////////////////////////////////////////////
//                 Menu options                  //
///////////////////////////////////////////////////

void call_back_function(int val){
  switch (val) {
    case 2:
      if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) + 0.15);
      else Scale_Factor -= 0.15;
      break;
    case 1:
      if(Scale_Factor < 1)Scale_Factor = 1/((1/Scale_Factor) - 0.15);
      else Scale_Factor += 0.15;
      break;
    case 3:
      Rotation_Factor -= 0.01*PI;
      break;
    case 4:
      Rotation_Factor += 0.01*PI;
      break;
    case 18:
      Equalization = 1;
      break;
    case 6:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = 1/9.0; Mask[1] = 1/9.0; Mask[2] = 1/9.0;
      Mask[3] = 1/9.0; Mask[4] = 1/9.0; Mask[5] = 1/9.0;
      Mask[6] = 1/9.0; Mask[7] = 1/9.0; Mask[8] = 1/9.0;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
      break;
    case 8:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = 1/16.0; Mask[1] = 2/16.0; Mask[2] = 1/16.0;
      Mask[3] = 2/16.0; Mask[4] = 4/16.0; Mask[5] = 2/16.0;
      Mask[6] = 1/16.0; Mask[7] = 2/16.0; Mask[8] = 1/16.0;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
      break;
    case 9:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = 0; Mask[1] = -1; Mask[2] = 0;
      Mask[3] = -1; Mask[4] = 4; Mask[5] = -1;
      Mask[6] = 0; Mask[7] = -1; Mask[8] = 0;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
      break;
    case 10:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = -1; Mask[1] = -1; Mask[2] = -1;
      Mask[3] = -1; Mask[4] = 8; Mask[5] = -1;
      Mask[6] = -1; Mask[7] = -1; Mask[8] = -1;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
      break;
    case 11:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = -1; Mask[1] = 0; Mask[2] = 1;
      Mask[3] = -1; Mask[4] = 0; Mask[5] = 1;
      Mask[6] = -1; Mask[7] = 0; Mask[8] = 1;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
    case 12:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = -1; Mask[1] = 0; Mask[2] = 1;
      Mask[3] = -2; Mask[4] = 0; Mask[5] = 2;
      Mask[6] = -1; Mask[7] = 0; Mask[8] = 1;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
      break;
    case 13:
      Flag_Filt = 1;
      Dim_Con = 3;
      Mask[0] = 1; Mask[1] = 2; Mask[2] = 1;
      Mask[3] = 0; Mask[4] = 0; Mask[5] = 0;
      Mask[6] = -1; Mask[7] = -2; Mask[8] = -1;
      cudaMemcpy(DMask, Mask, 9*sizeof(float), cudaMemcpyHostToDevice);
    case 5:
      Flag_PPnoise = 1;
      break;
    case 7:
      Flag_Med = 1;
      break;
    case 19:
      Flag_Fourier = 1;
      break;
    case 20:
      Flag_Reset = 1;
      Scale_Factor = 1;
      Rotation_Factor = 0;
      break;
    case 16:
      Flag_Ero = 1;
      Dim_Con = 3;
      Max_E = 255;
      break;
    case 17:
      Flag_Dil = 1;
      Dim_Con = 3;
      break;
    case 14:
      Flag_Gray = 1;
      break;
    case 15:
      Flag_BW = 1;
      break;
    case 21:
      exit(0);
      break;
    case 22:
      Flag_Pix = 1;
      break;
    case 23:
      Flag_req = 1;
      break;
    default:{
      }
    }
  display();
}

///////////////////////////////////////////////////
//                 Creating menu                 //
///////////////////////////////////////////////////
void Create_call_back_function(void) {
	sub_00 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Acercar", 1);
	glutAddMenuEntry("Alejar", 2);
        glutAddMenuEntry("Rotar derecha", 3);
        glutAddMenuEntry("Rotar izquierda", 4);

	sub_01 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Agregar ruido", 5);
	glutAddMenuEntry("Filtro de media", 6);
        glutAddMenuEntry("Filtro de mediana", 7);
        glutAddMenuEntry("Filtro gaussiano", 8);

	sub_02 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("2D - 4 conexion", 9);
	glutAddMenuEntry("2D - 8 conexion", 10);
        glutAddMenuEntry("Prewitt", 11);
        glutAddMenuEntry("Sobel X", 12);
        glutAddMenuEntry("Sobel Y", 13);

        sub_03 = glutCreateMenu(call_back_function);
	glutAddMenuEntry("Escala de grises", 14);
	glutAddMenuEntry("Binarizado", 15);

        sub_04 = glutCreateMenu(call_back_function);
        glutAddMenuEntry("Erosion", 16);
        glutAddMenuEntry("Dilatacion", 17);

        glutCreateMenu(call_back_function);
	glutAddMenuEntry("Pixelado", 22);
        glutAddMenuEntry("Recuantizacion", 23);
        glutAddSubMenu("Rotacion-escala", sub_00);
        glutAddSubMenu("Ruido-suavizado", sub_01);
	glutAddSubMenu("Deteccion de bordes", sub_02);
	glutAddSubMenu("Sistemas de color", sub_03);
        glutAddSubMenu("Operaciones morfologicas", sub_04);
        glutAddMenuEntry("Ecualizacion", 18);
        glutAddMenuEntry("Transformacion Fourier", 19);
        glutAddMenuEntry("Restaurar original", 20);
        glutAddMenuEntry("Salir", 21);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}


///////////////////////////////////////////////////
//  Struct and function to read data from image  //
///////////////////////////////////////////////////

typedef struct BMP_Info{
  unsigned long bytesInHeader;
  unsigned long widht;
  unsigned long height;
  unsigned int planes;
  unsigned int bitsPerPixel;
  unsigned long compression;
  unsigned long sizeImage;
  unsigned long hResolution;
  unsigned long vResolution;
  unsigned long nIndexes;
  unsigned long nIIndexes;
  char type[3];
  unsigned long size;
  char reserved[5];
  unsigned long offset;
} BMP_Info;

unsigned long Turn_Data_Long(FILE* fp){
  uint32_t data32;
  fread (&(data32),4, 1,fp);
  unsigned long data = (unsigned long)data32;
  return data;
}

unsigned int Turn_Data_Int(FILE* fp){
  uint16_t data16;
  fread (&(data16), 2, 1, fp);
  unsigned int data = (unsigned int)data16;
  return data;
}

void Read_Image(FILE* fp, BMP_Info* Image_Raw){
  fgets(Image_Raw->type, 3, fp);
  Image_Raw->size = Turn_Data_Long(fp);
  fgets(Image_Raw->reserved, 5, fp);
  Image_Raw->offset = Turn_Data_Long(fp);
  Image_Raw->bytesInHeader = Turn_Data_Long(fp);
  Image_Raw->widht = Turn_Data_Long(fp);
  Image_Raw->height = Turn_Data_Long(fp);
  Image_Raw->planes = Turn_Data_Int(fp);
  Image_Raw->bitsPerPixel = Turn_Data_Int(fp);
  Image_Raw->compression = Turn_Data_Long(fp);
  Image_Raw->sizeImage = Turn_Data_Long(fp);
  Image_Raw->hResolution = Turn_Data_Long(fp);
  Image_Raw->vResolution = Turn_Data_Long(fp);
  Image_Raw->nIndexes = Turn_Data_Long(fp);
  Image_Raw->nIIndexes = Turn_Data_Long(fp);
}

FILE *fp;
BMP_Info Image_Raw;
