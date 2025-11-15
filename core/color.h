#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <stdio.h>

// TODO: change all instances of calling vec3 for color
#define color_create vec3_create

typedef vec3 color;

void write_color(FILE* out, color pixel_color);
static inline double linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return sqrt(linear_component);

  return 0;
}
#endif
