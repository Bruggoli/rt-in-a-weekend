#ifndef SOLID_COLOR_H
#define SOLID_COLOR_H 

#include "texture.h"


typedef struct {
  color albedo;
} solid_color;

texture* solid_color_albedo(color albedo);
texture* solid_color_rgb(double red, double green, double blue);

#endif
