#include "translate.h"
#include "../accel/aabb.h"
#include "ray.h"
#include <stdlib.h>


hittable* translate_obj(hittable* object, vec3 offset) {

  fprintf(stderr, "\n=== TRANSLATE ===\n");
  fprintf(stderr, "Offset: (%.2f, %.2f, %.2f)\n", offset.e[0], offset.e[1], offset.e[2]);
  
  aabb bbox_before = object->bounding_box(object);
  fprintf(stderr, "Before - X: [%.2f, %.2f] Y: [%.2f, %.2f] Z: [%.2f, %.2f]\n",
          bbox_before.x.min, bbox_before.x.max,
          bbox_before.y.min, bbox_before.y.max,
          bbox_before.z.min, bbox_before.z.max);

  hittable* h = malloc(sizeof(hittable));
  translate* t = malloc(sizeof(translate));
  t->object = object;
  t->offset = offset;

  aabb bbox = object->bounding_box(object);
  t->bbox = aabb_offset(bbox, offset);
  

  h->data = t;
  h->hit = translate_hit_fn;
  h->bounding_box = translate_bounding_box; 
  aabb bbox_after = h->bounding_box(h);
  fprintf(stderr, "After  - X: [%.2f, %.2f] Y: [%.2f, %.2f] Z: [%.2f, %.2f]\n",
          bbox_after.x.min, bbox_after.x.max,
          bbox_after.y.min, bbox_after.y.max,
          bbox_after.z.min, bbox_after.z.max);
  
  return h;
}


bool translate_hit_fn(hittable* self, ray r, interval ray_t, hit_record* rec) {
  translate* t = (translate*)self->data;
  // move ray backwards by offset
  ray offset_r = ray_create(vec3_sub(r.orig, t->offset), r.dir, r.tm);

  if (!t->object->hit(t->object, offset_r, ray_t, rec))
    return false;

  rec->p = vec3_add(rec->p, t->offset);
  return true;
}

aabb translate_bounding_box(hittable* self) {
  translate* t = (translate*)self->data;
  return t->bbox;
}
