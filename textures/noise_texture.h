#ifndef NOISE_TEXTURE_H
#define NOISE_TEXTURE_H
#include "perlin.h"
#include "texture.h"
#include "../core/color.h"

typedef struct {
  perlin noise;
  double scale;
} noise_texture;

texture* noise_texture_create(double scale);

color noise_texture_value(texture* self, double u, double v, point3 p);

#endif
