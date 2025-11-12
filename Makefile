CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3 -fopenmp
# GPU architecture - automatically detect or use default
# Common values: sm_52 (Maxwell), sm_61 (Pascal), sm_75 (Turing), sm_86 (Ampere)
# To override: make GPU_ARCH=sm_75
GPU_ARCH ?= sm_52
NVCCFLAGS = -O3 -arch=$(GPU_ARCH) --use_fast_math --compiler-options "-O3 -fPIC"
LDFLAGS = -lm -lcudart -L/usr/local/cuda/lib64 -fopenmp
TARGET = main

# C source files
C_SOURCES = $(filter-out camera_cuda.c, $(wildcard *.c))
C_OBJECTS = $(C_SOURCES:.c=.o)

# C++ sources (denoiser - optional)
CXX_SOURCES = $(wildcard *.cpp)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

# CUDA source files (using ultra-optimized version)
CU_SOURCES = camera_cuda_optimized.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Optional: Enable OIDN denoising (requires Intel Open Image Denoise)
# To enable: make ENABLE_OIDN=1
ENABLE_OIDN ?= 0

ifeq ($(ENABLE_OIDN),1)
    CXXFLAGS += -DHAVE_OIDN
    LDFLAGS += -lOpenImageDenoise
    $(info Building WITH Intel OIDN denoising support)
else
    $(info Building WITHOUT denoising (set ENABLE_OIDN=1 to enable))
endif

all: $(TARGET)

$(TARGET): $(C_OBJECTS) $(CXX_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(C_OBJECTS) $(CXX_OBJECTS) $(CU_OBJECTS) $(LDFLAGS) -o $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -O3 -c $< -o $@

%.o: %.cu
	@echo "Compiling CUDA with architecture: $(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CXX_OBJECTS) $(CU_OBJECTS) $(TARGET) image.ppm *.o

run: $(TARGET)
	./$(TARGET) > image.ppm

# Auto-detect GPU architecture
detect:
	@echo "Detecting GPU architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//g' | sed 's/^/sm_/'

.PHONY: all clean run detect
