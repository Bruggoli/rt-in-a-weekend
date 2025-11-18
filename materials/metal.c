#include "metal.h"
#include "../core/hittable.h"
#include "material.h"
#include "../core/ray.h"
#include "../core/vec3.h"
#include <stdlib.h>

bool metal_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered);

material* mat_metal(color albedo, double fuzz) {
  material* m = malloc(sizeof(material));
  metal* met = malloc(sizeof(metal));
  met->albedo = albedo;
  met->fuzz = fuzz;

  m->data = met;
  m->scatter = metal_scatter;
  m->emit = material_emitted_default;

  return m;
}


bool metal_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered) {
  vec3 reflected = reflect(r_in.dir, rec->normal);
  metal* m = (metal*)self->data;
  reflected = vec3_add(unit_vector(reflected), vec3_scale(vec3_random_unit_vector(), m->fuzz));
  *scattered = ray_create(rec->p, reflected, r_in.tm);
  *attenuation = m->albedo;
  return true;
}
