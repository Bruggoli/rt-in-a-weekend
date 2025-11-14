#include "hittable.h"
#include "vec3.h"

void set_face_normal(hit_record *rec, ray r, vec3 outward_normal){
  rec->front_face = vec3_dot(r.dir, outward_normal) < 0;
  rec->normal = rec->front_face ? outward_normal : vec3_negate(outward_normal);
}
