#import "aabb.h"
#import "interval.h"
#import "hittable.h"


aabb aabb_create_from_interval(interval x, interval y, interval z) {
  
  fprintf(stderr, "creating aabb from interval...\n");
  aabb bb;

  bb.x = x;
  bb.y = y;
  bb.z = z;

  fprintf(stderr, "created aabb from interval!\n");
  return bb;
}

aabb aabb_create_from_point(point3 a, point3 b) {
  fprintf(stderr, "creating aabb from point...\n");
  aabb bb;

  bb.x = (a.e[0] <= b.e[0] ? interval_create(a.e[0], b.e[0]) : interval_create(b.e[0], a.e[0]));
  bb.y = (a.e[1] <= b.e[1] ? interval_create(a.e[1], b.e[1]) : interval_create(b.e[1], a.e[1]));
  bb.z = (a.e[2] <= b.e[2] ? interval_create(a.e[2], b.e[2]) : interval_create(b.e[2], a.e[2]));

  fprintf(stderr, "created aabb from point!\n");
  return bb;
}


aabb aabb_add(aabb box0, aabb box1) {

  aabb box_out;
  box_out.x = interval_enclose(box0.x, box1.x);
  box_out.y = interval_enclose(box0.y, box1.y);
  box_out.z = interval_enclose(box0.z, box1.z);
  return box_out;
}

interval axis_interval(aabb* bb, int n) {
  if (n == 1) return bb->y;
  if (n == 2) return bb->z;
  return bb->x;
}

bool aabb_hit(aabb* bb ,ray r, interval ray_t) {


  point3  ray_orig  = r.orig;
  vec3    ray_dir   = r.dir;

  for (int axis = 0; axis < 3; axis++) {
    interval ax = axis_interval(bb, axis);
    double adinv = 1.0 / ray_dir.e[axis];

    double t0 = (ax.min - ray_orig.e[axis]) * adinv; 
    double t1 = (ax.max - ray_orig.e[axis]) * adinv;

    if (t0 < t1) {
      if (t0 > ray_t.min) ray_t.min = t0;
      if (t1 < ray_t.max) ray_t.max = t1;
    } else {
      if (t1 > ray_t.min) ray_t.min = t1;
      if (t0 < ray_t.max) ray_t.max = t0;
    }

    if (ray_t.max <= ray_t.min)
      return false;

  }

  return true;
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
