#include "color.h"
#include "vec3.h"
#include <stdio.h>

typedef vec3 color;

void write_color(FILE *out, color pixel_color) {
  double r = pixel_color.e[0];
  double g = pixel_color.e[1];
  double b = pixel_color.e[2];

  int rbyte = (int)255.999 * r;
  int gbyte = (int)255.999 * g;
  int bbyte = (int)255.999 * b;

  fprintf(out, "%d %d %d\n", rbyte, gbyte, bbyte); 
}
