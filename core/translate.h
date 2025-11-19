#ifndef TRANSLATE_H
#define TRANSLATE_H

#include "hittable.h"

typedef struct {
  hittable* object;
  vec3 offset;
  aabb bbox;
} translate;

hittable* translate_obj(hittable* object, vec3 offset);

bool translate_hit_fn(hittable* self, ray r, interval ray_t, hit_record* rec);

aabb translate_bounding_box(hittable* self);

#endif
