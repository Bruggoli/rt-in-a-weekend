#ifndef TEXTURE_H
#define TEXTURE_H

#include "color.h"
#include "vec3.h"

typedef struct {
  color albedo;
} texture;

texture solid_color_albedo(color albedo);
texture solid_color_rgb(double red, double green, double blue);

color texture_value(double u, double v, point3 p);


#endif
