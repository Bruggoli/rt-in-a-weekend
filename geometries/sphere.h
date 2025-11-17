#ifndef SPHERE_H
#define SPHERE_H

#include "../core/hittable.h"
#include "../core/vec3.h"
#include <stdbool.h>

typedef struct {
  ray center;
  vec3 velocity;
  double radius;
  material* mat;
  aabb bbox;
} sphere;

hittable* sphere_create(point3 center, double radius, material* mat);
hittable* sphere_create_moving(point3 center1, point3 center2, double radius, material* mat);
void get_sphere_uv(point3 p, double* u, double* v);
aabb sphere_bounding_box(hittable* self);
void sphere_destroy(hittable* h);

#endif
