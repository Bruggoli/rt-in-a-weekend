#ifndef MATERIAL_H
#define MATERIAL_H

#include "../core/vec3.h"
#include "../core/color.h"
#include "../core/hittable.h"

typedef struct material material;

typedef bool (*scatter_fn)(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered);

bool material_scatter_default(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered);

typedef color (*emitted_fn)(material* self, double u, double v, point3 p);

color material_emitted_default(material* self, double u, double v, point3 p);

struct material {
  scatter_fn scatter;
  emitted_fn emit;
  void* data;
};


#endif
