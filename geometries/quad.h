#ifndef QUAD_H
#define QUAD_H

#include "../core/hittable.h"

typedef struct {
  point3 Q;
  vec3 u,
       v,
       w,
       normal;
  material* mat;
  aabb bbox;
  double D;
} quad;

hittable* quad_create(point3 Q, vec3 u, vec3 v, material* mat);

void quad_set_bounding_box(quad* self);

aabb quad_bounding_box(hittable* self);

bool quad_hit(hittable* self, ray r, interval ray_t, hit_record* rec);

bool is_interior(double a, double b, hit_record* rec);

#endif
