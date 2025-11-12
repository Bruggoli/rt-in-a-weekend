#include "bvh.h"
#include "camera_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>

// Helper to get min/max
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }

// Compute AABB for a sphere
AABB sphere_aabb(vec3 center, double radius) {
    AABB box;
    box.min.e[0] = center.e[0] - radius;
    box.min.e[1] = center.e[1] - radius;
    box.min.e[2] = center.e[2] - radius;
    box.max.e[0] = center.e[0] + radius;
    box.max.e[1] = center.e[1] + radius;
    box.max.e[2] = center.e[2] + radius;
    return box;
}

// Compute AABB containing two AABBs
AABB surrounding_box(AABB box1, AABB box2) {
    AABB box;
    box.min.e[0] = dmin(box1.min.e[0], box2.min.e[0]);
    box.min.e[1] = dmin(box1.min.e[1], box2.min.e[1]);
    box.min.e[2] = dmin(box1.min.e[2], box2.min.e[2]);
    box.max.e[0] = dmax(box1.max.e[0], box2.max.e[0]);
    box.max.e[1] = dmax(box1.max.e[1], box2.max.e[1]);
    box.max.e[2] = dmax(box1.max.e[2], box2.max.e[2]);
    return box;
}

// Surface area of AABB
static double aabb_surface_area(AABB box) {
    double dx = box.max.e[0] - box.min.e[0];
    double dy = box.max.e[1] - box.min.e[1];
    double dz = box.max.e[2] - box.min.e[2];
    return 2.0 * (dx * dy + dy * dz + dz * dx);
}

// Get centroid of AABB
static vec3 aabb_centroid(AABB box) {
    vec3 c;
    c.e[0] = (box.min.e[0] + box.max.e[0]) * 0.5;
    c.e[1] = (box.min.e[1] + box.max.e[1]) * 0.5;
    c.e[2] = (box.min.e[2] + box.max.e[2]) * 0.5;
    return c;
}

// Sphere info for BVH construction
typedef struct {
    AABB bounds;
    vec3 centroid;
    int index;
} SphereInfo;

// Compare functions for sorting
static int compare_x(const void* a, const void* b) {
    SphereInfo* sa = (SphereInfo*)a;
    SphereInfo* sb = (SphereInfo*)b;
    if (sa->centroid.e[0] < sb->centroid.e[0]) return -1;
    if (sa->centroid.e[0] > sb->centroid.e[0]) return 1;
    return 0;
}

static int compare_y(const void* a, const void* b) {
    SphereInfo* sa = (SphereInfo*)a;
    SphereInfo* sb = (SphereInfo*)b;
    if (sa->centroid.e[1] < sb->centroid.e[1]) return -1;
    if (sa->centroid.e[1] > sb->centroid.e[1]) return 1;
    return 0;
}

static int compare_z(const void* a, const void* b) {
    SphereInfo* sa = (SphereInfo*)a;
    SphereInfo* sb = (SphereInfo*)b;
    if (sa->centroid.e[2] < sb->centroid.e[2]) return -1;
    if (sa->centroid.e[2] > sb->centroid.e[2]) return 1;
    return 0;
}

// BVH Builder context
typedef struct {
    BVHNode* nodes;
    int node_count;
    int node_capacity;
    CudaSphere* spheres;
    int* sphere_indices;  // Reordered sphere indices
} BVHBuilder;

// Allocate new BVH node
static int allocate_node(BVHBuilder* builder) {
    if (builder->node_count >= builder->node_capacity) {
        builder->node_capacity *= 2;
        builder->nodes = realloc(builder->nodes, builder->node_capacity * sizeof(BVHNode));
    }
    return builder->node_count++;
}

// Recursively build BVH using SAH (Surface Area Heuristic)
static int build_recursive(
    BVHBuilder* builder,
    SphereInfo* infos,
    int start,
    int count
) {
    int node_idx = allocate_node(builder);
    BVHNode* node = &builder->nodes[node_idx];

    // Compute bounds of all spheres
    AABB bounds = infos[start].bounds;
    for (int i = 1; i < count; i++) {
        bounds = surrounding_box(bounds, infos[start + i].bounds);
    }
    node->bounds = bounds;

    // Leaf node (4 or fewer spheres)
    if (count <= 4) {
        node->sphere_start = start;
        node->sphere_count = count;
        node->left_child = -1;
        node->right_child = -1;

        // Copy sphere indices
        for (int i = 0; i < count; i++) {
            builder->sphere_indices[start + i] = infos[start + i].index;
        }

        return node_idx;
    }

    // Interior node - find best split using SAH
    AABB centroid_bounds;
    centroid_bounds.min = infos[start].centroid;
    centroid_bounds.max = infos[start].centroid;

    for (int i = 1; i < count; i++) {
        vec3 c = infos[start + i].centroid;
        centroid_bounds.min.e[0] = dmin(centroid_bounds.min.e[0], c.e[0]);
        centroid_bounds.min.e[1] = dmin(centroid_bounds.min.e[1], c.e[1]);
        centroid_bounds.min.e[2] = dmin(centroid_bounds.min.e[2], c.e[2]);
        centroid_bounds.max.e[0] = dmax(centroid_bounds.max.e[0], c.e[0]);
        centroid_bounds.max.e[1] = dmax(centroid_bounds.max.e[1], c.e[1]);
        centroid_bounds.max.e[2] = dmax(centroid_bounds.max.e[2], c.e[2]);
    }

    // Find longest axis
    double dx = centroid_bounds.max.e[0] - centroid_bounds.min.e[0];
    double dy = centroid_bounds.max.e[1] - centroid_bounds.min.e[1];
    double dz = centroid_bounds.max.e[2] - centroid_bounds.min.e[2];

    int axis = 0;
    if (dy > dx && dy > dz) axis = 1;
    else if (dz > dx && dz > dy) axis = 2;

    // Sort along chosen axis
    if (axis == 0) {
        qsort(&infos[start], count, sizeof(SphereInfo), compare_x);
    } else if (axis == 1) {
        qsort(&infos[start], count, sizeof(SphereInfo), compare_y);
    } else {
        qsort(&infos[start], count, sizeof(SphereInfo), compare_z);
    }

    // Use SAH to find optimal split point
    const int num_buckets = 12;
    double min_cost = DBL_MAX;
    int best_split = count / 2;

    // Try different split positions
    for (int i = 1; i < count; i++) {
        AABB left_box = infos[start].bounds;
        for (int j = 1; j < i; j++) {
            left_box = surrounding_box(left_box, infos[start + j].bounds);
        }

        AABB right_box = infos[start + i].bounds;
        for (int j = i + 1; j < count; j++) {
            right_box = surrounding_box(right_box, infos[start + j].bounds);
        }

        double cost = i * aabb_surface_area(left_box) +
                     (count - i) * aabb_surface_area(right_box);

        if (cost < min_cost) {
            min_cost = cost;
            best_split = i;
        }
    }

    // Split and recurse
    node->sphere_start = -1;
    node->sphere_count = 0;
    node->left_child = build_recursive(builder, infos, start, best_split);
    node->right_child = build_recursive(builder, infos, start + best_split, count - best_split);

    return node_idx;
}

// Build BVH from sphere array
BVHNode* build_bvh(void* sphere_array, int num_spheres, int* out_node_count) {
    double start_time = omp_get_wtime();
    fprintf(stderr, "Building BVH for %d spheres using %d CPU threads...\n",
            num_spheres, omp_get_max_threads());

    CudaSphere* spheres = (CudaSphere*)sphere_array;

    // Create sphere info array (parallelized)
    SphereInfo* infos = malloc(num_spheres * sizeof(SphereInfo));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_spheres; i++) {
        infos[i].bounds = sphere_aabb(spheres[i].center, spheres[i].radius);
        infos[i].centroid = aabb_centroid(infos[i].bounds);
        infos[i].index = i;
    }

    // Initialize builder
    BVHBuilder builder;
    builder.node_capacity = num_spheres * 2;  // Worst case
    builder.nodes = malloc(builder.node_capacity * sizeof(BVHNode));
    builder.node_count = 0;
    builder.spheres = spheres;
    builder.sphere_indices = malloc(num_spheres * sizeof(int));

    // Build tree (recursive - cannot be parallelized easily)
    double tree_start = omp_get_wtime();
    build_recursive(&builder, infos, 0, num_spheres);
    double tree_time = omp_get_wtime() - tree_start;

    // Reorder spheres based on BVH traversal order (parallelized)
    double reorder_start = omp_get_wtime();
    CudaSphere* reordered_spheres = malloc(num_spheres * sizeof(CudaSphere));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_spheres; i++) {
        reordered_spheres[i] = spheres[builder.sphere_indices[i]];
    }
    memcpy(spheres, reordered_spheres, num_spheres * sizeof(CudaSphere));
    double reorder_time = omp_get_wtime() - reorder_start;

    double total_time = omp_get_wtime() - start_time;
    fprintf(stderr, "✓ BVH built: %d nodes for %d spheres in %.3f sec (tree: %.3f, reorder: %.3f)\n",
            builder.node_count, num_spheres, total_time, tree_time, reorder_time);

    free(infos);
    free(reordered_spheres);
    free(builder.sphere_indices);

    *out_node_count = builder.node_count;
    return builder.nodes;
}

void free_bvh(BVHNode* nodes) {
    free(nodes);
}
