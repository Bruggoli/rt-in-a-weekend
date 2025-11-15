#include "solid_color.h"
#include "../core/color.h"
#include <stdlib.h>


color solid_color_value(texture* self, double u, double v, point3 p);

texture* solid_color_albedo(color albedo) {
  texture* tx = malloc(sizeof(texture));
  solid_color* sc = malloc(sizeof(solid_color));
  
  sc->albedo = albedo;
  
  tx->value = solid_color_value;
  tx->data  = sc;
  return tx;
}

texture* solid_color_rgb(double red, double green, double blue) {
  return solid_color_albedo(color_create(red, green, blue));
}

color solid_color_value(texture* self, double u, double v, point3 p) {
  solid_color* sc = self->data;
  return sc->albedo;
}


