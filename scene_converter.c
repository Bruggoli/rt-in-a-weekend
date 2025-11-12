#include "scene_converter.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "lambertian.h"
#include "metal.h"
#include "dielectric.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Helper to identify material type
static CudaMaterialType identify_material_type(material* mat) {
    // We need to identify the material by checking which scatter function it uses
    // This is a bit of a hack but works with the current architecture

    // Create temporary materials to compare function pointers
    material* test_lambertian = mat_lambertian(vec3_create(0, 0, 0));
    material* test_metal = mat_metal(vec3_create(0, 0, 0), 0);
    material* test_dielectric = mat_dielectric(1.0);

    CudaMaterialType type;

    if (mat->scatter == test_lambertian->scatter) {
        type = CUDA_LAMBERTIAN;
    } else if (mat->scatter == test_metal->scatter) {
        type = CUDA_METAL;
    } else if (mat->scatter == test_dielectric->scatter) {
        type = CUDA_DIELECTRIC;
    } else {
        type = CUDA_LAMBERTIAN; // default
    }

    free(test_lambertian);
    free(test_metal);
    free(test_dielectric);

    return type;
}

// Convert scene to CUDA-compatible format
void convert_scene_to_cuda(
    hittable* world,
    CudaSphere** out_spheres,
    int* out_num_spheres,
    CudaMaterial** out_materials,
    int* out_num_materials
) {
    hittable_list* list = (hittable_list*)world->data;

    // First pass: count spheres and unique materials
    int num_spheres = list->count;

    // Allocate temporary arrays
    material** material_ptrs = malloc(num_spheres * sizeof(material*));
    int* material_indices = malloc(num_spheres * sizeof(int));

    // Collect all material pointers
    for (int i = 0; i < num_spheres; i++) {
        sphere* s = (sphere*)list->objects[i]->data;
        material_ptrs[i] = s->mat;
    }

    // Find unique materials
    material** unique_materials = malloc(num_spheres * sizeof(material*));
    int num_materials = 0;

    for (int i = 0; i < num_spheres; i++) {
        bool found = false;
        for (int j = 0; j < num_materials; j++) {
            if (material_ptrs[i] == unique_materials[j]) {
                material_indices[i] = j;
                found = true;
                break;
            }
        }
        if (!found) {
            unique_materials[num_materials] = material_ptrs[i];
            material_indices[i] = num_materials;
            num_materials++;
        }
    }

    // Allocate output arrays
    CudaSphere* spheres = malloc(num_spheres * sizeof(CudaSphere));
    CudaMaterial* materials = malloc(num_materials * sizeof(CudaMaterial));

    // Convert spheres
    for (int i = 0; i < num_spheres; i++) {
        sphere* s = (sphere*)list->objects[i]->data;
        spheres[i].center = s->center;
        spheres[i].radius = s->radius;
        spheres[i].material_idx = material_indices[i];
    }

    // Convert materials
    for (int i = 0; i < num_materials; i++) {
        material* mat = unique_materials[i];
        CudaMaterialType type = identify_material_type(mat);

        materials[i].type = type;

        if (type == CUDA_LAMBERTIAN) {
            lambertian* lam = (lambertian*)mat->data;
            materials[i].albedo = lam->albedo;
            materials[i].fuzz = 0.0;
            materials[i].ref_idx = 0.0;
        } else if (type == CUDA_METAL) {
            metal* met = (metal*)mat->data;
            materials[i].albedo = met->albedo;
            materials[i].fuzz = met->fuzz;
            materials[i].ref_idx = 0.0;
        } else if (type == CUDA_DIELECTRIC) {
            dielectric* die = (dielectric*)mat->data;
            materials[i].albedo = vec3_create(1.0, 1.0, 1.0);
            materials[i].fuzz = 0.0;
            materials[i].ref_idx = die->refraction_index;
        }
    }

    // Set output
    *out_spheres = spheres;
    *out_num_spheres = num_spheres;
    *out_materials = materials;
    *out_num_materials = num_materials;

    // Cleanup
    free(material_ptrs);
    free(material_indices);
    free(unique_materials);

    fprintf(stderr, "Converted %d spheres and %d materials for CUDA\n", num_spheres, num_materials);
}

void free_cuda_scene(CudaSphere* spheres, CudaMaterial* materials) {
    free(spheres);
    free(materials);
}
