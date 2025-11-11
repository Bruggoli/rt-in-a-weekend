#include "ray.h"
#include "vec3.h"

ray ray_create(point3 origin, vec3 direction){
  ray r;
  r.orig = origin;
  r.dir = direction;
  return r;
}

point3 ray_origin(ray r){
  return r.orig;
}

vec3 ray_direction(ray r){
  return r.dir;
}

point3 ray_at(ray r, double t){
  return vec3_add(r.orig, vec3_scale(r.dir, t));
}

void ray_print(FILE* out, ray r) {
  vec3_print(stderr, r.dir);
  vec3_print(stderr, r.orig);
}
