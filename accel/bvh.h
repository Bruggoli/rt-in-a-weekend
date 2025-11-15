#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "../core/hittable.h"
#include "../core/hittable_list.h"
#include <stddef.h>

typedef struct {
  hittable* left;
  hittable* right;
  aabb bbox;
} bvh_node;

hittable* bvh_node_create_range(hittable_list* list);

hittable* bvh_node_create(hittable** objects, size_t start, size_t end);

bool bvh_node_hit(hittable* self, ray r, interval ray_t, hit_record* rec);

int box_compare(const void* a, const void* b, int axis_index);

aabb bvh_node_bounding_box(hittable* self);

int box_x_compare(const void* a, const void* b); 
int box_y_compare(const void* a, const void* b); 
int box_z_compare(const void* a, const void* b); 



#endif
