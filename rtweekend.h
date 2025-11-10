#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#define INFINITY_VAL INFINITY
#define PI 3.1415926535897932385

double degrees_to_radians(double degres) {
  return degres * PI / 180.0;
}

// Common headers
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"

#endif
