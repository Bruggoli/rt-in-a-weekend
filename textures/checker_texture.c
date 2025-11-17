#include "checker_texture.h"
#include "solid_color.h"
#include "../utils/rtw_stb_image.h"
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

color checker_texture_value_fn(texture* self, double u, double v, point3 p) {
  checker_texture* tx = self->data;
  int x = (int)floor(tx->inv_scale * p.e[0]);
  int y = (int)floor(tx->inv_scale * p.e[1]);
  int z = (int)floor(tx->inv_scale * p.e[2]);
   

  bool isEven = (x + y + z) % 2 == 0;

  // returns the texture of even if isEven, odd if not
  return isEven ? tx->even->value(tx->even, u, v, p) : tx->odd->value(tx->odd, u, v, p);
}
