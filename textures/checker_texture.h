#ifndef CHECKER_TEXTURE_H
#define CHECKER_TEXTURE_H
#include "texture.h"

typedef struct {
  double inv_scale;
  texture* even;
  texture* odd;
} checker_texture;

texture* checker_texture_create_texture(double scale, texture* even, texture* odd);

texture* checker_texture_create_color(double scale, color c1, color c2);

color checker_texture_value_fn(texture* tx, double u, double v, point3 p);

#endif
