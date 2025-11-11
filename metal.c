#include "metal.h"
#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"
#include <stdlib.h>

bool metal_scatter(material* self, ray r_in, hit_record*, color* attenuation, ray* scattered);

material* mat_metal(color albedo) {
  material* m = malloc(sizeof(material));
  metal* met = malloc(sizeof(metal));
  met->albedo = albedo;

  m->data = met;
  m->scatter = metal_scatter;

  return m;
}


bool metal_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered) {
  fprintf(stderr, "Before reflect: ");
  vec3 reflected = reflect(r_in.dir, rec->normal);
  fprintf(stderr, "After reflect: ");
  vec3_print(stderr, reflected);
  *scattered = ray_create(rec->p, reflected);
  metal* m = (metal*)self->data;
  *attenuation = m->albedo;
  return true;
}
