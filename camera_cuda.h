#ifndef CAMERA_CUDA_H
#define CAMERA_CUDA_H

#include "vec3.h"
#include "camera.h"

#ifdef __cplusplus
extern "C" {
#endif

// CUDA-compatible structures (must match camera_cuda.cu)
// Note: These use vec3 (double precision) on host side
// They are converted to float precision inside CUDA code for performance
typedef struct {
    vec3 center;
    double radius;
    int material_idx;
} CudaSphere;

typedef enum {
    CUDA_LAMBERTIAN = 0,
    CUDA_METAL = 1,
    CUDA_DIELECTRIC = 2
} CudaMaterialType;

typedef struct {
    CudaMaterialType type;
    vec3 albedo;
    double fuzz;      // for metal
    double ref_idx;   // for dielectric
} CudaMaterial;

// CUDA render function
void cuda_render(
    vec3* host_image_buffer,
    void* host_world,
    camera* cam,
    CudaSphere* host_spheres,
    int num_spheres,
    CudaMaterial* host_materials,
    int num_materials
);

#ifdef __cplusplus
}
#endif

#endif // CAMERA_CUDA_H
