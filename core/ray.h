#ifndef RAY_H
#define RAY_H

#include "vec3.h"

typedef struct {
  point3 orig;
  vec3 dir;
  double tm;
} ray;

ray ray_create(point3 origin, vec3 direction, double time);
point3 ray_origin(ray r);
vec3 ray_direction(ray r);

static inline point3 ray_at(ray r, double t){
  return vec3_add(r.orig, vec3_scale(r.dir, t));
}
void ray_print(FILE* out, ray r);
double time(ray r);

#endif
