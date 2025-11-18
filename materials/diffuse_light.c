#include "diffuse_light.h"
#include "../textures/solid_color.h"
#include <stdlib.h>


material* diffuse_light_create_texture(texture* tx) {
  material* m = malloc(sizeof(material));
  diffuse_light* dl = malloc(sizeof(diffuse_light));

  dl->tex = tx;

  m->data = dl;
  m->emit = diffuse_light_emitted;
  m->scatter = material_scatter_default;
  return m;
}

material* diffuse_light_create_color(color emit) {
  return diffuse_light_create_texture(solid_color_albedo(emit));
}

color diffuse_light_emitted(material* self, double u, double v, point3 p) {
  diffuse_light* dl = self->data;
  return dl->tex->value(dl->tex, u, v, p);
}
