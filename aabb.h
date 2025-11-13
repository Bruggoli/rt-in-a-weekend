#ifndef AABB_H
#define AABB_H
#import "interval.h"
#import "vec3.h"
#import "ray.h"

// looks ugly but circular dependencies
typedef struct {

}hittable;

typedef struct {
  interval  x,
            y,
            z;
} aabb;

aabb aabb_create_from_interval(interval x, interval y, interval z);
aabb aabb_create_from_point(point3 a, point3 b);
aabb aabb_create_empty();

aabb aabb_add(aabb box0, aabb box1);

interval axis_interval(aabb* bb, int n);

int longest_axis(hittable* self);

bool aabb_hit(aabb* bb, ray r, interval ray_t);

void aabb_print(FILE* out, aabb bb);

#endif
