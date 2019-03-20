CC = nvcc
CFLAGS = -lglut -lGLEW -lGL -lm -ccbin clang-3.8 -lstdc++
SRCS = main.cu
PROG = out

build: main.cu
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)
run:
	./$(PROG)
clear:
	rm -rf out
