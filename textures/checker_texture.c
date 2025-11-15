#include "checker_texture.h"
#include "solid_color.h"
#include <stdlib.h>


texture* checker_texture_create_texture(double scale, texture* even, texture* odd) {
  texture* tx = malloc(sizeof(texture));
  checker_texture* ct = malloc(sizeof(checker_texture));
  
  ct->inv_scale = 1.0 / scale;
  ct->even      = even;
  ct->odd       = odd;

  tx->data = ct;
  tx->value = checker_texture_value_fn;

  return tx;
}

texture* checker_texture_create_color(double scale, color c1, color c2) {
  return checker_texture_create_texture(scale, solid_color_albedo(c1), solid_color_albedo(c2));
}

color checker_texture_value_fn(texture* self, double u, double v, point3 p);
