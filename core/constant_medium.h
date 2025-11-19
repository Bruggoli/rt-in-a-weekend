#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.h"
#include "../materials/material.h"
#include "../textures/texture.h"

typedef struct {
  hittable* boundary;
  double neg_inv_density;
  material* phase_function;
} constant_medium;


hittable* constant_medium_create_texture(hittable* boundary, double density, texture* tx);

hittable* constant_medium_create_color(hittable* boundary, double density, color albedo);

bool constant_medium_hit(hittable* self, ray r, interval ray_t, hit_record* rec);

aabb constant_medium_bbox_fn(hittable* self);



#endif
