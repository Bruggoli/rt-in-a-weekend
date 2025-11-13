#include "hittable_list.h"
#include "aabb.h"
#include "hittable.h"
#include "rtweekend.h"

#include <stdlib.h>

#define INITIAL_CAPACITY 10

hittable* hittable_list_create(void) {
  hittable* h = malloc(sizeof(hittable));
  hittable_list* list = malloc(sizeof(hittable_list));
  list->capacity = INITIAL_CAPACITY;
  list->count = 0;
  // Creates a memory allocation based on the size of one hittable* 
  // instance, then multiplies that by max capacity
  list->objects = malloc(sizeof(hittable*) * list->capacity);

  h->data = list;
  h->hit = hittable_list_hit;

  return h;
}

void hittable_list_add(hittable* h, hittable* object) {
  hittable_list* list = (hittable_list*) h->data;
  // checks if cap is full
  // if it is double capacity
  if (list->capacity <= list->count) {
    list->capacity *= 2;
    list->objects = realloc(list->objects, sizeof(hittable*) * list->capacity);
  }

  h->bounding_box = aabb_add(h->bounding_box, object->bounding_box);
  // Adds the objects to the list, incrementing count after
  list->objects[list->count++] = object;
}

void hittable_list_clear(hittable_list *list) {
  // TODO: should probably free objects from memory
  list->count = 0;
}

bool hittable_list_hit(hittable* self, ray r, interval ray_t, hit_record *rec) {
  // data is a void pointer
  // this means the compiler doesnt know what data is
  // for the compiler to understand we have to type-cast it
  hittable_list* list = (hittable_list*)self->data;
  hit_record temp_rec;
  bool hit_anything = false;
  double closests_so_far = ray_t.max;

  for (int i = 0; i < list->count; i++) {
    if (list->objects[i]->hit(list->objects[i], r, interval_create(ray_t.min, closests_so_far), &temp_rec)) {
      hit_anything = true;
      closests_so_far = temp_rec.t;
      *rec = temp_rec;
    }
  }

  return hit_anything;
}

void hittable_list_destroy(hittable_list *list) {
  free(list->objects);
  free(list);
}
