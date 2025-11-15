#ifndef MATERIAL_H
#define MATERIAL_H

#include "../core/vec3.h"
#include "../core/color.h"
#include "../core/hittable.h"

typedef struct material material;

typedef bool (*scatter_fn)(material* self, ray r_in, hit_record* rec, vec3* attenuation, ray* scattered);


struct material {
  scatter_fn scatter;
  void* data;
};


#endif
