#include "aabb.h"
#include "../core/interval.h"

static void pad_to_minimums(aabb* bbox);

aabb aabb_create_from_interval(interval x, interval y, interval z) {
  
  aabb bb;

  bb.x = x;
  bb.y = y;
  bb.z = z;

  pad_to_minimums(&bb);
  return bb;
}

aabb aabb_create_from_point(point3 a, point3 b) {
  aabb bb;

  bb.x = (a.e[0] <= b.e[0] ? interval_create(a.e[0], b.e[0]) : interval_create(b.e[0], a.e[0]));
  bb.y = (a.e[1] <= b.e[1] ? interval_create(a.e[1], b.e[1]) : interval_create(b.e[1], a.e[1]));
  bb.z = (a.e[2] <= b.e[2] ? interval_create(a.e[2], b.e[2]) : interval_create(b.e[2], a.e[2]));

  pad_to_minimums(&bb);
  return bb;
}


aabb aabb_add(aabb box0, aabb box1) {

  aabb box_out;
  box_out.x = interval_enclose(box0.x, box1.x);
  box_out.y = interval_enclose(box0.y, box1.y);
  box_out.z = interval_enclose(box0.z, box1.z);
  return box_out;
}


int longest_axis(aabb* bb) {
  double x = interval_size(bb->x);
  double y = interval_size(bb->y);
  double z = interval_size(bb->z);
  if (x > z)
    return x > z ? 0 : 2;
  else
    return y > z ? 1 : 2;
}

void aabb_print(FILE* out, aabb bb) {
  fprintf(out, "AABB: x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]\n",
          bb.x.min, bb.x.max,
          bb.y.min, bb.y.max,
          bb.z.min, bb.z.max);
}

static void pad_to_minimums(aabb* bbox) {
  double delta = 0.0001;
  if (interval_size(bbox->x) < delta) bbox->x = interval_expand(bbox->x, delta);
  if (interval_size(bbox->y) < delta) bbox->y = interval_expand(bbox->y, delta);
  if (interval_size(bbox->z) < delta) bbox->z = interval_expand(bbox->z, delta);
    
}
