#ifndef DIELECTRIC_H
#define DIELECTRIC_H
#include "../core/hittable.h"
#include "../core/color.h"

typedef struct {
  double refraction_index;
} dielectric;


material* mat_dielectric(double refraction_index);

bool dielectric_scatter(material* self, ray r, hit_record* rec, color* attenuation, ray* scattered);

double reflectance(double cosine, double refraction_index);

#endif
