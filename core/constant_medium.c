#include "constant_medium.h"
#include "../utils/rtweekend.h"
#include "interval.h"
#include "../materials/isotropic.h"
#include <stdlib.h>

hittable* constant_medium_create_texture(hittable* boundary, double density, texture* tx) {
  hittable* h = malloc(sizeof(hittable));
  constant_medium* cm = malloc(sizeof(constant_medium));

  cm->boundary = boundary;
  cm->neg_inv_density = -1/density;
  cm->phase_function = isotropic_create_texture(tx);

  h->data = cm;
  h->hit = constant_medium_hit;
  h->bounding_box = constant_medium_bbox_fn;

  return h;
}

hittable* constant_medium_create_color(hittable* boundary, double density, color albedo) {
  hittable* h = malloc(sizeof(hittable));
  constant_medium* cm = malloc(sizeof(constant_medium)); // TODO: implement isotropic


  cm->boundary = boundary;
  cm->neg_inv_density = -1/density;
  cm->phase_function = isotropic_create_color(albedo);

  h->data = cm;
  h->hit = constant_medium_hit;
  h->bounding_box = constant_medium_bbox_fn;

  return h;
}

bool constant_medium_hit(hittable* self, ray r, interval ray_t, hit_record* rec) {
  constant_medium* cm = self->data;
  hit_record rec1, rec2;

  if (!cm->boundary->hit(cm->boundary, r, interval_create(-INFINITY, INFINITY), &rec1)){
    return false;
  }
  
  if (!cm->boundary->hit(cm->boundary, r, interval_create(rec1.t+0.0001, INFINITY), &rec2)){
    return false;
  }
  
  if (rec1.t < ray_t.min)
    rec1.t = ray_t.min;

  if (rec2.t > ray_t.max)
    rec2.t = ray_t.max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;



  double ray_length = vec3_length(r.dir);
  double distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
  double hit_distance = cm->neg_inv_density * log(random_double());

  if (hit_distance > distance_inside_boundary)
    return false;

  rec->t = rec1.t + hit_distance / ray_length;
  rec->p = ray_at(r, rec->t);
  rec->mat = cm->phase_function;
  rec->normal = vec3_create(1, 0, 0);  // Arbitrary
  rec->front_face = true;              // Arbitrary
  
  return true;
}

aabb constant_medium_bbox_fn(hittable* self) {
  constant_medium* cm = (constant_medium*)self->data;
  return cm->boundary->bounding_box(cm->boundary);
}
