#include "vec3.h"
#include "rtweekend.h"
#include <math.h>
#include <stdbool.h>


// constructor
/// vec3.e[0] = x
/// vec3.e[1] = y
/// vec3.e[2] = z
///
vec3 vec3_create(double x, double y, double z) {
  return (vec3){x, y, z};
}

// Operations
vec3 vec3_add(vec3 u, vec3 v){
  return vec3_create(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
vec3 vec3_sub(vec3 u, vec3 v){
  return vec3_create(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
vec3 vec3_mul(vec3 u, vec3 v){
  return vec3_create(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
} 
/// vec3 u, double t
/// returns new scaled vec3
vec3 vec3_scale(vec3 u, double t){
  return vec3_create(u.e[0] * t, u.e[1] * t, u.e[2] * t);
} 
vec3 vec3_div(vec3 u, double t){
  return vec3_create(u.e[0] / t, u.e[1] / t, u.e[2] / t);
}

vec3 vec3_cross(vec3 u, vec3 v){
  return vec3_create(
    u.e[1] * v.e[2] - u.e[2] * v.e[1], 
    u.e[2] * v.e[0] - u.e[0] * v.e[2], 
    u.e[0] * v.e[1] - u.e[1] * v.e[0]
  );
}

vec3 vec3_negate(vec3 v) {
  return vec3_create(-v.e[0],-v.e[1],-v.e[2]);
}

double vec3_dot(vec3 u, vec3 v) {
  return  u.e[0] * v.e[0] + 
          u.e[1] * v.e[1] +
          u.e[2] * v.e[2];
}

double vec3_length_squared(vec3 v) {
  return v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2];
}

double vec3_length(vec3 v) {
  return sqrt(vec3_length_squared(v));
}

vec3 unit_vector(vec3 v) {
  return vec3_scale(v, 1.0 / vec3_length(v));
}

vec3 vec3_random_on_hemisphere(vec3 normal) {
  vec3 on_unit_sphere = vec3_random_unit_vector();
  if (vec3_dot(on_unit_sphere, normal))
    return on_unit_sphere;
  else
   return vec3_negate(on_unit_sphere);
}

vec3 reflect(vec3 v, vec3 n){
  double dot_vn = vec3_dot(v, n);
  vec3 scaled_n = vec3_scale(n, 2.0 * dot_vn);
  return vec3_sub(v, scaled_n);
}

vec3 vec3_random_unit_vector() {
  while(true) {
    vec3 p = vec3_random_range(-1, 1);
    double lensq = vec3_length_squared(p);
    if (1e-160 < lensq && lensq <= 1)
      return vec3_div(p, sqrt(lensq));
  }
}

bool near_zero(vec3 v) {
  double s = 1e-8;
  return (fabs(v.e[0]) < s) && (fabs(v.e[1]) < s) && (fabs(v.e[2]) < s);
}

vec3 vec3_random() {
  return vec3_create(random_double(), random_double(), random_double());
}

vec3 vec3_random_range(double min, double max) {
  return vec3_create(random_double_range(min, max), random_double_range(min, max), random_double_range(min, max));
}

void vec3_print(FILE* out, vec3 v) {
    fprintf(out, "%f %f %f\n", v.e[0], v.e[1], v.e[2]);
}

