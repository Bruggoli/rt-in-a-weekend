#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "../core/color.h"
#include "../core/hittable.h"

typedef struct {
  color albedo;
} lambertian;

material* mat_lambertian(color albedo);

#endif
