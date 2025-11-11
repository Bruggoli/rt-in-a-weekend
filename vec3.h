#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>
#include <math.h>
#include <stdio.h>

typedef struct{
  double e[3];
} vec3;

typedef vec3 point3;

// constructor
vec3 vec3_create(double x, double y, double z);

// Operations
vec3 vec3_add(vec3 u, vec3 v);

vec3 vec3_sub(vec3 u, vec3 v);

vec3 vec3_mul(vec3 u, vec3 v);

/// vec3 u, double t
/// returns new scaled vec3
vec3 vec3_scale(vec3 u, double t);

vec3 vec3_div(vec3 u, double t);

vec3 vec3_cross(vec3 u, vec3 v);

vec3 vec3_negate(vec3 v);

vec3 unit_vector(vec3 v);

double vec3_dot(vec3 u, vec3 v);

double vec3_length(vec3 v);

double vec3_length_squared(vec3 v);

vec3 vec3_unit(vec3 v);

vec3 vec3_random_unit_vector();

vec3 vec3_random_on_hemisphere(vec3 normal);

bool near_zero(vec3 v);

vec3 reflect(vec3 v, vec3 n);

vec3 vec3_random();

vec3 vec3_random_range(double min, double max);

void vec3_print(FILE* out, vec3 v);



#endif
