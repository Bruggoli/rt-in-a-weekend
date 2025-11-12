#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include <stdbool.h>

typedef struct {
  point3 center;
  vec3 velocity;
  double radius;
  material* mat;
} sphere;

hittable* sphere_create(point3 center, double radius, material* mat);
hittable* sphere_create_moving(point3 center1, point3 center2, double radius, material* mat);
void sphere_destroy(hittable* h);

#endif
