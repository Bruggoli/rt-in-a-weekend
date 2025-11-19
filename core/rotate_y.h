#ifndef ROTATE_Y_H
#define ROTATE_Y_H
#include "hittable.h"
#include "../accel/aabb.h"

typedef struct {
  hittable* object;
  double sin_theta,
         cos_theta;
  aabb bbox;
} rotate_y;

hittable* rotate_object_y(hittable* object, double angle);

bool rotated_y_hit_fn(hittable* self, ray r, interval ray_t, hit_record* rec);

aabb rotated_bounding_box_fn(hittable* self);


#endif
