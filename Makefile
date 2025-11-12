CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3
NVCCFLAGS = -O3 -arch=sm_70 --compiler-options -fPIC
LDFLAGS = -lm -lcudart -L/usr/local/cuda/lib64
TARGET = main

# C source files (excluding camera.c which will be compiled separately)
C_SOURCES = $(filter-out camera_cuda.c, $(wildcard *.c))
C_OBJECTS = $(C_SOURCES:.c=.o)

# CUDA source files
CU_SOURCES = camera_cuda.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(C_OBJECTS) $(CU_OBJECTS) $(LDFLAGS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CU_OBJECTS) $(TARGET) image.ppm

run: $(TARGET)
	./$(TARGET) > image.ppm

.PHONY: all clean run
