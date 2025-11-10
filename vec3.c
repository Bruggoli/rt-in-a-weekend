#include "vec3.h"


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
