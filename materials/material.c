#include "material.h"

color material_emitted_default(material* self, double u, double v, point3 p) {
  return color_create(0, 0, 0);  // Black - no emission
}

bool material_scatter_default(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered) {
  return false;
}
