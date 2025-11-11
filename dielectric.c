#include "dielectric.h"
#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

bool dielectric_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered);

material* mat_dielectric(double refraction_index){
  
  material* m = malloc(sizeof(material));
  dielectric* d = malloc(sizeof(dielectric));
 
  d->refraction_index = refraction_index;
  m->data = d;
  m->scatter = dielectric_scatter;

  return m;
}

bool dielectric_scatter(material* self, ray r_in, hit_record* rec, color* attenuation, ray* scattered) {
  *attenuation = vec3_create(1.0, 1.0, 1.0);
  // cast data to dielectric to access variables of the "class"
  dielectric* d = (dielectric*)self->data;
  double ri = rec->front_face ? (1.0 / d->refraction_index) : d->refraction_index;

  vec3 unit_direction = unit_vector(r_in.dir);
  double cos_theta = fmin(vec3_dot(vec3_negate(unit_direction), rec->normal), 1.0);
  double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

  bool cannot_refract = ri * sin_theta > 1.0;
  vec3 direction;

  if (cannot_refract || reflectance(cos_theta, ri) > random_double()) 
    direction = reflect(unit_direction, rec->normal);
  else
   direction = refract(unit_direction, rec->normal, ri);

  vec3 refracted = refract(unit_direction, rec->normal, ri);

  *scattered = ray_create(rec->p, direction);

  return true;

}

double reflectance(double cosine, double refraction_index) {
  double r0 = (1 - refraction_index) / (1 + refraction_index);
  r0 = r0 * r0;
  return r0 + (1-r0) * pow((1 - cosine), 5);
}
