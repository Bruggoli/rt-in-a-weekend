#include "sphere.h"
#include "hittable.h"
#include "ray.h"
#include "vec3.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

// forward declaration
// needs to be declared here before its used in the "constructor" and the 
// args arent applicable to every other shape i might implement
bool sphere_hit(hittable* self, ray r, interval ray_t, hit_record* rec);


hittable* sphere_create(point3 center, double radius, material* mat) {
  // TODO: Initialise pointer to the material `mat`
  hittable* h = malloc(sizeof(hittable));
  sphere* s = malloc(sizeof(sphere));

  s->center = center;
  s->radius = fmax(0, radius);
  s->mat    = mat;

  h->data = s;
  h->hit = sphere_hit;

  return h;
};

bool sphere_hit(hittable* self, ray r, interval ray_t, hit_record* rec) {
  sphere* s = (sphere*)self->data;
  vec3 oc = vec3_sub(s->center, r.orig);
  double a = vec3_length_squared(r.dir);
  double h = vec3_dot(r.dir, oc);
  double c = vec3_length_squared(oc) - s->radius * s->radius;

  double discriminant = h*h - a*c;
  double sqrtd = sqrt(discriminant);


  if (discriminant < 0) {
    return false;
  }

  double root = (h - sqrtd) / a;
  if (root <= ray_t.min || ray_t.max <= root) {
    root = (h + sqrtd) / a;
    if (root <= ray_t.min || ray_t.max <= root){
      return false;
    }
  }

  rec->t = root;
  rec->p = ray_at(r, root);
  vec3 outward_normal = vec3_div(vec3_sub(rec->p, s->center), s->radius);
  set_face_normal(rec, r, outward_normal); 
  rec->mat = s->mat;

  return true;
}

void sphere_destroy(hittable* h) {
  free(h->data);
  free(h);
}

