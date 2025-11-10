#ifndef HITTABLE_LIST
#define HITTABLE_LIST

#include "hittable.h"

typedef struct {
  hittable ** objects; // pointer to an array of pointers (idk how it works yet)
  int count;
  int capacity;
} hittable_list;

hittable* hittable_list_create(void);
void hittable_list_add(hittable* h, hittable* object);
void hittable_list_clear(hittable_list* list);
bool hittable_list_hit(hittable* self, ray r, double ray_tmin, double ray_tmax, hit_record* rec);
void hittable_list_destroy(hittable_list* list);

#endif
