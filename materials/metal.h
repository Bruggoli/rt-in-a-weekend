#ifndef METAL_H 
#define METAL_H

#include "color.h"
#include "hittable.h"

typedef struct {
  color albedo;
  double fuzz;
} metal;

material* mat_metal(color albedo, double fuzz);

#endif
