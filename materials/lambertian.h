#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "../core/color.h"
#include "../core/hittable.h"
#include "../textures/texture.h"
#include "../textures/solid_color.h"

typedef struct {
  texture* tx;
} lambertian;

material* mat_lambertian_tx(texture* tx);
material* mat_lambertian(color albedo);

#endif
