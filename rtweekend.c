#include "rtweekend.h"
#include <stdlib.h>

double degrees_to_radians(double degres) {
  return degres * PI / 180.0;
}

double random_double() {
  return random() / (RAND_MAX + 1.0);
}

double random_double_range(double min, double max) {
  return min + (max-min)*random_double();
}
