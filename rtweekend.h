#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define INFINITY_VAL INFINITY
#define PI 3.1415926535897932385

double degrees_to_radians(double degres);

double random_double();
double random_double_range(double min, double max);


// Common headers
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "interval.h"

#endif
