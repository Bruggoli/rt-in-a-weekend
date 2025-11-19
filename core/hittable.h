#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "vec3.h"
#include "interval.h"
#include <stdbool.h>
#include "../accel/aabb.h"

// forward declaration to prevent recursive include calls
typedef struct material material;

typedef struct hit_record {
  point3 p;
  vec3 normal;
  double  t,
          u,
          v;
  bool front_face;
  material* mat;

} hit_record;

void set_face_normal(hit_record* rec, ray r, vec3 outward_normal); 

typedef struct hittable hittable;

typedef bool (*hit_fn)(hittable* self, ray r, interval ray_t, hit_record* rec);

typedef aabb (*bounding_box_fn)(hittable* self);

struct hittable {
  hit_fn hit;
  bounding_box_fn bounding_box;
  void* data;
};


#endif
