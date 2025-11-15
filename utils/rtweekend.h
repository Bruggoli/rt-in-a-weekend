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
int random_int(int min, int max);


// Common headers
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/color.h"
#include "../core/hittable.h"
#include "../core/interval.h"

#endif
