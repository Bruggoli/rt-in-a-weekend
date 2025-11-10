#include "color.h"
#include "vec3.h"
#include "interval.h"
#include <stdio.h>

typedef vec3 color;

void write_color(FILE *out, color pixel_color) {
  double r = pixel_color.e[0];
  double g = pixel_color.e[1];
  double b = pixel_color.e[2];

  interval intensity = interval_create(0.000, 0.999);

  int rbyte = (int)256 * clamp(intensity, r);
  int gbyte = (int)256 * clamp(intensity, g);
  int bbyte = (int)256 * clamp(intensity, b);


  fprintf(out, "%d %d %d\n", rbyte, gbyte, bbyte); 
}
