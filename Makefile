CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3
# GPU architecture - automatically detect or use default
# Common values: sm_52 (Maxwell), sm_61 (Pascal), sm_75 (Turing), sm_86 (Ampere)
# To override: make GPU_ARCH=sm_75
GPU_ARCH ?= sm_52
NVCCFLAGS = -O3 -arch=$(GPU_ARCH) --use_fast_math --compiler-options "-O3 -fPIC"
LDFLAGS = -lm -lcudart -L/usr/local/cuda/lib64
TARGET = main

# C source files (excluding camera.c which will be compiled separately)
C_SOURCES = $(filter-out camera_cuda.c, $(wildcard *.c))
C_OBJECTS = $(C_SOURCES:.c=.o)

# CUDA source files (using BVH-accelerated version)
CU_SOURCES = camera_cuda_bvh.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(C_OBJECTS) $(CU_OBJECTS) $(LDFLAGS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	@echo "Compiling CUDA with architecture: $(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CU_OBJECTS) $(TARGET) image.ppm *.o

run: $(TARGET)
	./$(TARGET) > image.ppm

# Auto-detect GPU architecture
detect:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//g' | sed 's/^/sm_/'

.PHONY: all clean run detect
