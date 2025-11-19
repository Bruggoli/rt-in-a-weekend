#ifndef ISOTROPIC_H
#define ISOTROPIC_H

#include "material.h"
#include "../textures/texture.h"

typedef struct {
  texture* tx;
} isotropic;

material* isotropic_create_color(color albedo);
material* isotropic_create_texture(texture* tx);

bool isotropic_scatter(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered);


#endif
