#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "color.h"
#include "hittable.h"

typedef struct {
  color albedo;
} lambertian;

material* mat_lambertian(color albedo);

#endif
