#include "sphere.h"
#include "../core/hittable.h"
#include "../core/ray.h"
#include "../core/vec3.h"
#include "../accel/aabb.h"
#include "../utils/rtweekend.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

// forward declaration
// needs to be declared here before its used in the "constructor" and the 
// args arent applicable to every other shape i might implement
bool sphere_hit(hittable* self, ray r, interval ray_t, hit_record* rec);
aabb sphere_bounding_box(hittable* self);


hittable* sphere_create(point3 static_center, double radius, material* mat) {
  hittable* h = malloc(sizeof(hittable));
  sphere* s = malloc(sizeof(sphere));

  vec3 rvec = vec3_create(radius, radius, radius);
  s->bbox = aabb_create_from_point(vec3_sub(static_center, rvec), vec3_add(static_center, rvec));

  ray center = ray_create(static_center, vec3_create(0, 0, 0), 0);

  s->center = center;
  s->radius = fmax(0, radius);
  s->mat    = mat;

  h->data = s;
  h->bounding_box = sphere_bounding_box;
  h->hit = sphere_hit;

  return h;
};


hittable* sphere_create_moving(point3 center1, point3 center2, double radius, material* mat) {
  hittable* h = malloc(sizeof(hittable));
  sphere* s = malloc(sizeof(sphere));

  vec3 rvec = vec3_create(radius, radius, radius);
  ray center = ray_create(center1, vec3_sub(center2, center1), 0);
  aabb box1 = aabb_create_from_point(vec3_sub(ray_at(center, 0), rvec), vec3_add(ray_at(center, 0), rvec));
  aabb box2 = aabb_create_from_point(vec3_sub(ray_at(center, 1), rvec), vec3_add(ray_at(center, 1), rvec));

  s->center         = center;
  s->velocity       = vec3_sub(center2, center1);
  s->radius         = fmax(0, radius);
  s->mat            = mat;
  s->bbox           = aabb_add(box1, box2);

  h->data = s;
  h->hit  = sphere_hit;
  h->bounding_box = sphere_bounding_box;

  return h;
};

aabb sphere_bounding_box(hittable* self) {
  sphere* s = self->data;
  return s->bbox;
}

void get_sphere_uv(point3 p, double* u, double* v) {
  double theta = acos(-p.e[1]);
  double phi = atan2(-p.e[2], p.e[0]) + PI;

  *u = phi / (2*PI);
  *v = theta / PI;
}

bool sphere_hit(hittable* self, ray r, interval ray_t, hit_record* rec) {
  sphere* s = (sphere*)self->data;

  point3 current_center = ray_at(s->center, r.tm);
  vec3 oc = vec3_sub(current_center, r.orig);
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
  vec3 outward_normal = vec3_div(vec3_sub(rec->p, current_center), s->radius);
  set_face_normal(rec, r, outward_normal); 
  get_sphere_uv(outward_normal, &rec->u, &rec->v);
  rec->mat = s->mat;

  return true;
}

void sphere_destroy(hittable* h) {
  free(h->data);
  free(h);
}

