#include "quad.h"
#include <stdlib.h>

hittable* quad_create(point3 Q, vec3 u, vec3 v, material* mat) {
  hittable* h = malloc(sizeof(hittable));
  quad* q = malloc(sizeof(quad));

  q->Q = Q;
  q->u = u;
  q->v = v;
  q->mat = mat;

  vec3 n = vec3_cross(u, v);
  q->normal = unit_vector(n);
  q->D = vec3_dot(q->normal, q->Q);
  q->w = vec3_div(n, vec3_dot(n, n));

  quad_set_bounding_box(q);

  h->bounding_box = quad_bounding_box;
  h->hit = quad_hit;
  h->data = q;
  return h;
}

void quad_set_bounding_box(quad* self) {
  aabb bbox_diagonal1 = aabb_create_from_point(self->Q, vec3_add(vec3_add(self->Q, self->u), self->v));
  aabb bbox_diagonal2 = aabb_create_from_point(vec3_add(self->Q, self->u), vec3_add(self->Q, self->v));

  self->bbox = aabb_add(bbox_diagonal1, bbox_diagonal2);
}

aabb quad_bounding_box(hittable* self) {
  quad* q = (quad*)self->data;
  return q->bbox;
}

bool quad_hit(hittable* self, ray r, interval ray_t, hit_record* rec) {
  quad* q = (quad*)self->data;
  double denom = vec3_dot(q->normal, r.dir);

  // No hit if the ray is parallel to the plane
  if (fabs(denom) < 1e-8)
    return false;

  // return false if the hit point paramter t is outside the ray interval
  double t = (q->D - vec3_dot(q->normal, r.orig)) / denom;
  if (!interval_contains(ray_t, t))
    return false;

  // Determine if the hit point lies within the planar shape using its plane coordinates
  point3 intersection = ray_at(r, t);
  vec3 planar_hitpt_vector = vec3_sub(intersection, q->Q);
  double alpha = vec3_dot(q->w, vec3_cross(planar_hitpt_vector, q->v));
  double beta = vec3_dot(q->w, vec3_cross(q->u, planar_hitpt_vector));

  if (!is_interior(alpha, beta, rec))
    return false;

  rec->t = t;
  rec->p = intersection;
  rec->mat = q->mat;
  set_face_normal(rec, r, q->normal);

  return true;
}

bool is_interior(double a, double b, hit_record* rec) {
  interval unit_interval = interval_create(0, 1);
  // Given the hit point in plane coordinates, return false if it is outside the
  // primitive, otherwise set the hit record UV coordinates and return true

  if (!interval_contains(unit_interval, a) || !interval_contains(unit_interval, b))
    return false;

  rec->u = a;
  rec->v = b;
  return true;
}
