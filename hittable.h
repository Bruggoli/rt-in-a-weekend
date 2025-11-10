#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "vec3.h"
#include "interval.h"
#include <stdbool.h>

typedef struct hit_record {
  point3 p;
  vec3 normal;
  double t;
  bool front_face;

} hit_record;

void set_face_normal(hit_record* rec, ray r, vec3 outward_normal); 

typedef struct hittable hittable;

typedef bool (*hit_fn)(hittable* self, ray r, interval ray_t, hit_record* rec);

struct hittable {
  hit_fn hit;
  void* data;
};



#endif
