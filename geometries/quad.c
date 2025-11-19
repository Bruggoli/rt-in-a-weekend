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

hittable* create_box(point3 a, point3 b, material* mat) {
  fprintf(stderr, "\n=== BOX CREATE ===\n");
  fprintf(stderr, "Corner A: (%.2f, %.2f, %.2f)\n", a.e[0], a.e[1], a.e[2]);
  fprintf(stderr, "Corner B: (%.2f, %.2f, %.2f)\n", b.e[0], b.e[1], b.e[2]);
  hittable* sides = hittable_list_create();

  point3 min = vec3_create(fmin(a.e[0], b.e[0]), fmin(a.e[1], b.e[1]), fmin(a.e[2], b.e[2]));
  point3 max = vec3_create(fmax(a.e[0], b.e[0]), fmax(a.e[1], b.e[1]), fmax(a.e[2], b.e[2]));

  fprintf(stderr, "Box Min: (%.2f, %.2f, %.2f)\n", min.e[0], min.e[1], min.e[2]);
  fprintf(stderr, "Box Max: (%.2f, %.2f, %.2f)\n", max.e[0], max.e[1], max.e[2]);
  fprintf(stderr, "Box Size: %.2f x %.2f x %.2f\n", 
          max.e[0] - min.e[0], max.e[1] - min.e[1], max.e[2] - min.e[2]);

  vec3 dx = vec3_create(max.e[0] - min.e[0], 0, 0);
  vec3 dy = vec3_create(0, max.e[1] - min.e[1], 0);
  vec3 dz = vec3_create(0, 0, max.e[2] - min.e[2]);

  hittable_list_add(sides, quad_create(vec3_create(min.e[0], min.e[1], max.e[2]), dx, dy, mat));              // front
  hittable_list_add(sides, quad_create(vec3_create(max.e[0], min.e[1], max.e[2]), vec3_negate(dz), dy, mat)); // right
  hittable_list_add(sides, quad_create(vec3_create(max.e[0], min.e[1], min.e[2]), vec3_negate(dx), dy, mat)); // back
  hittable_list_add(sides, quad_create(vec3_create(min.e[0], min.e[1], min.e[2]), dz, dy, mat));              // left
  hittable_list_add(sides, quad_create(vec3_create(min.e[0], max.e[1], max.e[2]), dx, vec3_negate(dz), mat)); // top
  hittable_list_add(sides, quad_create(vec3_create(min.e[0], min.e[1], min.e[2]), dx, dz, mat));              // bottom

  return sides;
}



