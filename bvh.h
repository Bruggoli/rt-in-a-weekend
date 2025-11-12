#ifndef BVH_H
#define BVH_H

#include "vec3.h"

// Axis-Aligned Bounding Box
typedef struct {
    vec3 min;
    vec3 max;
} AABB;

// BVH Node (GPU-friendly, 64 bytes aligned)
typedef struct {
    AABB bounds;           // 48 bytes (2 × vec3)
    int left_child;        // 4 bytes (index to left child, or -1)
    int right_child;       // 4 bytes (index to right child, or -1)
    int sphere_start;      // 4 bytes (index into sphere array, or -1 if interior node)
    int sphere_count;      // 4 bytes (number of spheres in leaf, or 0 if interior)
} BVHNode;

// Build BVH from sphere array (CPU-side)
// Returns array of BVH nodes and total node count
BVHNode* build_bvh(
    void* spheres,         // CudaSphere* array
    int num_spheres,
    int* out_node_count
);

// Compute AABB for a sphere
AABB sphere_aabb(vec3 center, double radius);

// Compute AABB containing multiple AABBs
AABB surrounding_box(AABB box1, AABB box2);

// Free BVH memory
void free_bvh(BVHNode* nodes);

#endif // BVH_H
