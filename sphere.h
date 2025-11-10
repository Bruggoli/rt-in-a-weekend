#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include <stdbool.h>

typedef struct {
  point3 center;
  double radius;
} sphere;

hittable* sphere_create(point3 center, double radius);
void sphere_destroy(hittable* h);

#endif
