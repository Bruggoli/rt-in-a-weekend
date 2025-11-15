#ifndef AABB_H
#define AABB_H
#include "../core/interval.h"
#include "../core/vec3.h"
#include "../core/ray.h"

// looks ugly but circular dependencies
typedef struct hittable hittable;

typedef struct {
  interval  x,
            y,
            z;
} aabb;

aabb aabb_create_from_interval(interval x, interval y, interval z);
aabb aabb_create_from_point(point3 a, point3 b);
aabb aabb_create_empty();

aabb aabb_add(aabb box0, aabb box1);


static inline interval axis_interval(aabb* bb, int n) {
  if (n == 1) return bb->y;
  if (n == 2) return bb->z;
  return bb->x;
}

int longest_axis(aabb* self);

static inline bool aabb_hit(aabb* bb ,ray r, interval ray_t) {


  point3  ray_orig  = r.orig;
  vec3    ray_dir   = r.dir;

  for (int axis = 0; axis < 3; axis++) {
    interval ax = axis_interval(bb, axis);
    double adinv = 1.0 / ray_dir.e[axis];

    double t0 = (ax.min - ray_orig.e[axis]) * adinv; 
    double t1 = (ax.max - ray_orig.e[axis]) * adinv;

    if (t0 < t1) {
      if (t0 > ray_t.min) ray_t.min = t0;
      if (t1 < ray_t.max) ray_t.max = t1;
    } else {
      if (t1 > ray_t.min) ray_t.min = t1;
      if (t0 < ray_t.max) ray_t.max = t0;
    }

    if (ray_t.max <= ray_t.min)
      return false;

  }

  return true;
}
void aabb_print(FILE* out, aabb bb);

#endif
