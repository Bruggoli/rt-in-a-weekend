#include "bvh.h"
#include "aabb.h"
#include "../core/interval.h"
#include <stdlib.h>
#include <stddef.h>



int box_x_compare(const void* a, const void* b); 
int box_y_compare(const void* a, const void* b); 
int box_z_compare(const void* a, const void* b); 

hittable* bvh_node_create(hittable** objects, size_t start, size_t end) {


  hittable* h = malloc(sizeof(hittable));
  bvh_node* n = malloc(sizeof(bvh_node));

  h->hit = bvh_node_hit;
  h->bounding_box = bvh_node_bounding_box;

  aabb bbox;
  for (size_t object_index=start; object_index < end; object_index++)
    bbox = aabb_add(bbox, objects[object_index]->bounding_box(objects[object_index]));

  int axis = longest_axis(&bbox);

  int (*comparator)(const void*, const void*) = 
                    (axis == 0) ? box_x_compare
                  : (axis == 1) ? box_y_compare
                                : box_z_compare;

  size_t object_span = end - start;

  if (object_span == 1) {
    n->left = n->right = objects[start];
  } else if (object_span == 2) {
    n->left = objects[start];
    n->right = objects[start + 1];
  } else {
    qsort(&objects[start], object_span, sizeof(hittable*), comparator);

    size_t mid = start + object_span / 2;
    n->left = bvh_node_create(objects, start, mid);
    n->right = bvh_node_create(objects, mid, end);
  }
  n->bbox = bbox;
  h->data = n;
  return h;

}

hittable* bvh_node_create_range(hittable_list* list) {
  return bvh_node_create(list->objects, 0, list->count);
}

bool bvh_node_hit(hittable* self, ray r, interval ray_t, hit_record* rec) {
  bvh_node* n = (bvh_node*)self->data;

  if (!aabb_hit(&n->bbox, r, ray_t))
    return false;

  bool hit_left = n->left->hit(n->left, r, ray_t, rec);
  bool hit_right = n->right->hit(n->right, r, interval_create(ray_t.min, hit_left ? rec->t : ray_t.max), rec);

  return hit_left || hit_right;
}

aabb bvh_node_bounding_box(hittable* self) {
  bvh_node* n = (bvh_node*)self->data;
  return n->bbox;
}

int box_compare(const void* a, const void* b, int axis_index) {

    // What's actually stored at these addresses?
  hittable* ha = *(hittable**)a;
  hittable* hb = *(hittable**)b;

  if (!ha->bounding_box || !hb->bounding_box) {
    fprintf(stderr, "  ERROR: NULL hittable->bounding_box!\n");
    return 0;
  }
  aabb a_bb = ha->bounding_box(ha);
  aabb b_bb = hb->bounding_box(hb);

  interval a_axis_interval = axis_interval(&a_bb, axis_index);
  interval b_axis_interval = axis_interval(&b_bb, axis_index);
  
  return (a_axis_interval.min < b_axis_interval.min) ? -1 : (a_axis_interval.min > b_axis_interval.min) ? 1 : 0;
}

int box_x_compare(const void* a, const void* b) {
  return box_compare(a, b, 0);
}

int box_y_compare(const void* a, const void* b) {
  return box_compare(a, b, 1);
}

int box_z_compare(const void* a, const void* b) {
  return box_compare(a, b, 2);
}
