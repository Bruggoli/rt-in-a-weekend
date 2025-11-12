#ifndef SCENE_CONVERTER_H
#define SCENE_CONVERTER_H

#include "hittable.h"
#include "camera_cuda.h"

// Convert the polymorphic scene to flat CUDA-compatible arrays
void convert_scene_to_cuda(
    hittable* world,
    CudaSphere** out_spheres,
    int* out_num_spheres,
    CudaMaterial** out_materials,
    int* out_num_materials
);

// Free the converted scene
void free_cuda_scene(CudaSphere* spheres, CudaMaterial* materials);

#endif // SCENE_CONVERTER_H
