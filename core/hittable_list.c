#include "hittable_list.h"
#include "../accel/aabb.h"
#include "hittable.h"
#include "interval.h"

#include <stdio.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 10

hittable* hittable_list_create(void) {
  fprintf(stderr, "Creating hittable list...\n");
  hittable* h = malloc(sizeof(hittable));
  hittable_list* list = malloc(sizeof(hittable_list));
  list->capacity = INITIAL_CAPACITY;
  list->count = 0;
  // Creates a memory allocation based on the size of one hittable* 
  // instance, then multiplies that by max capacity
  list->objects = malloc(sizeof(hittable*) * list->capacity);

  list->bbox = (aabb){interval_empty(), interval_empty(), interval_empty()};

  h->bounding_box = hittable_list_bounding_box;
  h->data = list;
  h->hit = hittable_list_hit;

  return h;
}

void hittable_list_add(hittable* h, hittable* object) {
  fprintf(stderr, "Adding to hittable_list...\n");
  hittable_list* list = (hittable_list*) h->data;

  // checks if cap is full
  // if it is double capacity
  if (list->capacity <= list->count) {

    fprintf(stderr, "Malloc ran out for hittable list:%d/%d adding more...\n", list->count, list->capacity);
    list->capacity *= 2;
    list->objects = realloc(list->objects, sizeof(hittable*) * list->capacity);
  }


  if (!object) {
    fprintf(stderr, "ERROR: object is NULL!\n");
    return;
  }

  if (!object->bounding_box) {
    fprintf(stderr, "ERROR: object->bounding_box is NULL!\n");
    return;
  }

  fprintf(stderr, "calling bounding_box\n");
  list->bbox = aabb_add(list->bbox, object->bounding_box(object));
  fflush(stderr); 
  fprintf(stderr, "Checkpoint!\n");
  // Adds the objects to the list, incrementing count after
  list->objects[list->count++] = object;
  fprintf(stderr, "Added to hittable_list!\n");
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

aabb hittable_list_bounding_box(hittable* self) {
  fprintf(stderr, "Creating hlbb..\n");
  hittable_list* list = (hittable_list*)self->data;

  return list->bbox;
}


void hittable_list_destroy(hittable_list *list) {
  free(list->objects);
  free(list);
}
