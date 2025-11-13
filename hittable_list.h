#ifndef HITTABLE_LIST
#define HITTABLE_LIST

#include "hittable.h"
#include "interval.h"
#include "aabb.h"
typedef struct {
  hittable ** objects; // pointer to an array of pointers (idk how it works yet)
  int count;
  int capacity;
  aabb bbox;
} hittable_list;

hittable* hittable_list_create(void);
void hittable_list_add(hittable* h, hittable* object);
void hittable_list_clear(hittable_list* list);
bool hittable_list_hit(hittable* self, ray r, interval ray_t, hit_record *rec);
aabb hittable_list_bounding_box(hittable* self);
void hittable_list_destroy(hittable_list* list);

#endif
