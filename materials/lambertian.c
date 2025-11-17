#include "lambertian.h"
#include "material.h"
#include "../core/ray.h"
#include "../core/vec3.h"
#include <stdlib.h>
#include <stdbool.h>

bool lambertian_scatter(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered);


material* mat_lambertian_tx(texture* tx) {
  material* m = malloc(sizeof(material));
  lambertian* l = malloc(sizeof(lambertian));
  l->tx = tx;

  m->data = l;
  m->scatter = lambertian_scatter;

  return m;
}

material* mat_lambertian(color albedo) {
  return mat_lambertian_tx(solid_color_albedo(albedo));
}

bool lambertian_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered) {
  vec3 scatter_direction = vec3_add(rec->normal, vec3_random_unit_vector());

  if (near_zero(scatter_direction))
    scatter_direction = rec->normal;

  // TODO: figure out why this has to be dereferenced before use
  *scattered = ray_create(rec->p, scatter_direction, r_in.tm);

  lambertian* l = (lambertian*)self->data;
  // Same here!
  *attenuation = l->tx->value(l->tx, rec->u, rec->v, rec->p);
  return true;
}
