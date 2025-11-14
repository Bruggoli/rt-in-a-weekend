#include "texture.h"

texture solid_color_albedo(color albedo);
texture solid_color_rgb(double red, double green, double blue);

color texture_value(double u, double v, point3 p);
