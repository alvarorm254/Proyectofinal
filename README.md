# Proyectofinal

The project folder contains multiples cuda functions used to compile a image editor. The GUI was developed using GLUT and in order to acelerate the processing the prgrams where parallelized using cuda 9.0.<br/>

## Execution<br/>

The compilation line is shown below:<br/>
nvcc main.cu -o out -lglut -lGLEW -lGL -lm -ccbin clang-3.8 -lstdc++<br/>
Finally execute the program with:<br/>
./out<br/>
The project contains the following processing:<br/>
-Image subsamplig(pixelated images).<br/>
-Image requantization.<br/>
-Scale (zoom in, zoom out).<br/>
-Rotations.<br/>
-Adding salt and pepper noise.<br/>
-Smothing filters(mean,median,gaussian).<br/>
-Edges detection(laplacian fiter 4,8 conectivity).<br/>
-Edges detection(Prewitt and sobel masks).<br/>
-Color models(grayscale)<br/>
-Automatic thresholding.<br/>
-Morphologic operations(erode and dilate).<br/>
-Histogram equalization(3D-2D).<br/>
-Fourier Transform.<br/>

Since the interface was designed using glut the pull-down menu appears with mouse event (right click) automatically loading the image and processing as requested.<br/>
Some results are shown below:<br/>
ORIGINAL:<br/>
![alt text](https://github.com/alvarorm254/Proyectofinal/blob/master/monta%C3%B1a.bmp)<br/>
REQUANTIZATION:<br/>
![alt text](https://github.com/alvarorm254/Proyectofinal/blob/master/req.png)<br/>
SUBSAMPLING:<br/>
![alt text](https://github.com/alvarorm254/Proyectofinal/blob/master/subsampling.png)<br/>
