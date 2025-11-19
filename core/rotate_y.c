#include "rotate_y.h"
#include "hittable.h"
#include "ray.h"
#include "../utils/rtweekend.h"
#include "../accel/aabb.h"

#include <stdlib.h>

hittable* rotate_object_y(hittable* object, double angle) {

  hittable* h = malloc(sizeof(hittable));
  rotate_y* ry = malloc(sizeof(rotate_y));

  double radians = degrees_to_radians(angle);
  ry->sin_theta = sin(radians);
  ry->cos_theta = cos(radians);
  ry->bbox      = object->bounding_box(object);
  ry->object    = object;

  point3 min = { INFINITY_VAL,  INFINITY_VAL,  INFINITY_VAL};
  point3 max = {-INFINITY_VAL, -INFINITY_VAL, -INFINITY_VAL};

  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        double x = i * ry->bbox.x.max + (1-i)*ry->bbox.x.min;
        double y = j * ry->bbox.y.max + (1-j)*ry->bbox.y.min;
        double z = k * ry->bbox.z.max + (1-k)*ry->bbox.z.min;

        double new_x =  ry->cos_theta * x + ry->sin_theta * z;
        double new_z = -ry->sin_theta * x + ry->cos_theta * z;

        float tester[3] = {new_x, y, new_z};

        for (int c = 0; c < 3; c++) {
          min.e[c] = fmin(min.e[c], tester[c]);
          max.e[c] = fmax(max.e[c], tester[c]);
        }
      }
    }
  }



  ry->bbox = aabb_create_from_point(min, max);

  h->bounding_box = rotated_bounding_box_fn;
  h->hit = rotated_y_hit_fn;
  h->data = ry;

  return h;
}

bool rotated_y_hit_fn(hittable* self, ray r, interval ray_t, hit_record* rec) {
  rotate_y* ry = (rotate_y*)self->data;

  // Transform the ray from world space to object space

  point3 origin = vec3_create(
    (r.orig.e[0] * ry->cos_theta) - (r.orig.e[2] * ry->sin_theta),
    r.orig.e[1],
    (r.orig.e[0] * ry->sin_theta) + (r.orig.e[2] * ry->cos_theta)
  );

  point3 direction = vec3_create(
    (r.dir.e[0] * ry->cos_theta) - (r.dir.e[2] * ry->sin_theta),
    r.dir.e[1],
    (r.dir.e[0] * ry->sin_theta) + (r.dir.e[2] * ry->cos_theta)
  );

  ray rotated_r = ray_create(origin, direction, r.tm);

  // Determine whether an intersection exists in object space (and if so, where)

  if (!ry->object->hit(ry->object, rotated_r, ray_t, rec))
    return false;

  // Transform the intersection from object space back to world space
  
  rec->p = vec3_create(
    (rec->p.e[0] * ry->cos_theta) + (rec->p.e[2] * ry->sin_theta),
    rec->p.e[1],
    (rec->p.e[0] * -ry->sin_theta) + (rec->p.e[2] * ry->cos_theta)
  );

  rec->normal = vec3_create(
    (rec->normal.e[0] * ry->cos_theta) + (rec->normal.e[2] * ry->sin_theta),
    rec->normal.e[1],
    (rec->normal.e[0] * -ry->sin_theta) + (rec->normal.e[2] * ry->cos_theta)
  );

  return true;
}

aabb rotated_bounding_box_fn(hittable* self){
  rotate_y* ry = (rotate_y*)self->data;
  return ry->bbox;
}
