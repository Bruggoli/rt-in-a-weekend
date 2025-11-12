# Ray Tracing with CUDA Acceleration

This project implements a GPU-accelerated ray tracer using CUDA, based on "Ray Tracing in a Weekend".

## Features

- **CUDA GPU Acceleration**: Massively parallel ray tracing on NVIDIA GPUs
- **Automatic Scene Conversion**: Converts C polymorphic scene to GPU-friendly flat arrays
- **Complete Material Support**: Lambertian, Metal, and Dielectric materials
- **Defocus Blur**: Depth of field effects
- **Anti-Aliasing**: Multiple samples per pixel with cuRAND

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 7.0 or higher (e.g., RTX 20-series, GTX 16-series, or newer)

### Software
- CUDA Toolkit 10.0 or later
- GCC or compatible C compiler
- Make build system

## Installation

### 1. Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

**Arch Linux:**
```bash
sudo pacman -S cuda
```

### 2. Set Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Verify Installation

```bash
nvcc --version
nvidia-smi
```

## Building

```bash
make clean
make
```

This will:
1. Compile all C source files with GCC
2. Compile CUDA kernel with NVCC
3. Link everything into the `main` executable

## Running

```bash
# Render to image.ppm
make run

# Or manually:
./main > image.ppm
```

The rendered image will be saved as `image.ppm` in PPM format.

## Architecture

### CUDA Implementation

The CUDA implementation uses a flattened data structure for efficient GPU processing:

1. **Scene Conversion** (`scene_converter.c`):
   - Converts polymorphic C structures to flat arrays
   - Separates spheres and materials into contiguous GPU memory
   - Maps material pointers to indices

2. **CUDA Kernel** (`camera_cuda.cu`):
   - Device-side vector math operations
   - Sphere intersection tests
   - Material scattering (Lambertian, Metal, Dielectric)
   - cuRAND for random number generation
   - Iterative ray tracing (avoids recursion limits)

3. **Memory Management**:
   - Host-side scene is converted to flat arrays
   - Data transferred to GPU via `cudaMemcpy`
   - Rendering done entirely on GPU
   - Results copied back to host for output

### Performance

Expected performance improvements over CPU:

| Resolution | Samples | CPU Time* | GPU Time** | Speedup |
|------------|---------|-----------|------------|---------|
| 1200x675   | 50      | ~2.5 min  | ~5-15 sec  | ~10-30x |
| 1920x1080  | 100     | ~15 min   | ~20-45 sec | ~20-45x |
| 3840x2160  | 100     | ~60 min   | ~1.5-3 min | ~20-40x |

*CPU: Modern multi-core processor with OpenMP
**GPU: NVIDIA RTX 3070 or equivalent

## Customization

### Adjust GPU Architecture

If you have a different GPU, modify the Makefile:

```makefile
# For RTX 30-series (Ampere):
NVCCFLAGS = -O3 -arch=sm_86 --compiler-options -fPIC

# For RTX 20-series (Turing):
NVCCFLAGS = -O3 -arch=sm_75 --compiler-options -fPIC

# For GTX 10-series (Pascal):
NVCCFLAGS = -O3 -arch=sm_61 --compiler-options -fPIC
```

Find your compute capability: https://developer.nvidia.com/cuda-gpus

### Adjust Render Settings

Edit `main.c`:

```c
cam.image_width       = 1920;     // Resolution
cam.samples_per_pixel = 100;      // Anti-aliasing quality
cam.max_depth         = 50;       // Ray bounce limit
cam.vfov              = 20;       // Field of view
cam.defocus_angle     = 0.6;      // Depth of field
```

## Troubleshooting

### Out of Memory Errors

If you get CUDA out-of-memory errors:
1. Reduce image resolution
2. Reduce samples per pixel
3. Check available GPU memory: `nvidia-smi`

### Compilation Errors

- **`nvcc: command not found`**: CUDA Toolkit not installed or not in PATH
- **`cannot find -lcudart`**: CUDA library path not set correctly
- **Architecture mismatch**: Update `-arch=sm_XX` in Makefile to match your GPU

### Runtime Errors

- **No CUDA-capable device**: No NVIDIA GPU detected
- **Kernel launch failed**: Try reducing render resolution or check GPU availability

## File Structure

```
.
├── camera_cuda.cu       # CUDA kernel implementation
├── camera_cuda.h        # CUDA interface header
├── scene_converter.c    # Scene conversion to CUDA format
├── scene_converter.h    # Scene converter header
├── camera.c            # Rendering coordinator
├── main.c              # Scene setup and entry point
├── Makefile            # CUDA build system
└── *.c/h               # Other ray tracer components
```

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## License

This is an educational implementation following Peter Shirley's "Ray Tracing in One Weekend" series.
