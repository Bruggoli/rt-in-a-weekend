#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct {
  point3 orig;
  vec3 dir;
} ray;

ray ray_create(point3 origin, vec3 direction);
point3 ray_origin(ray r);
vec3 ray_direction(ray r);
point3 ray_at(ray r, double t);

#endif
