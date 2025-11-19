#include "isotropic.h"
#include "../textures/solid_color.h"
#include "../core/vec3.h"
#include <stdlib.h>

material* isotropic_create_texture(texture* tx) {
  material* m = malloc(sizeof(material));
  isotropic* iso = malloc(sizeof(isotropic));
  
  iso->tx = tx;

  m->data = iso;
  m->emit = material_emitted_default;
  m->scatter = isotropic_scatter;

  return m;
}

material* isotropic_create_color(color albedo) {
  return isotropic_create_texture(solid_color_albedo(albedo));
}


bool isotropic_scatter(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered) {
  isotropic* iso = (isotropic*)self->data;
  
  ray sray = ray_create(rec->p, vec3_random_unit_vector(), r_in.tm);
  *scattered = sray; 

  vec3 attn = iso->tx->value(iso->tx, rec->u, rec->v, rec->p);
  *attenuation = attn;

  return true;
}


