#ifndef AABB_H
#define AABB_H
#import "interval.h"
#import "vec3.h"
#import "ray.h"

typedef struct {
  interval  x,
            y,
            z;
} aabb;

aabb aabb_create_from_interval(interval x, interval y, interval z);
aabb aabb_create_from_point(point3 a, point3 b);

aabb aabb_add(aabb box0, aabb box1);

interval axis_interval(aabb* bb, int n);

bool aabb_hit(aabb* bb, ray r, interval ray_t);



#endif
