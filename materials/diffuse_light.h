#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H

#include "../textures/texture.h"
#include "material.h"

typedef struct {
  texture* tex;
} diffuse_light;

material* diffuse_light_create_texture(texture* tx);

material* diffuse_light_create_color(color emit);

color diffuse_light_emitted(material* self, double u, double v, point3 p);


#endif
