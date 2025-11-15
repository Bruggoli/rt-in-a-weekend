#ifndef TEXTURE_H
#define TEXTURE_H

#include "../core/color.h"
#include "../core/vec3.h"

typedef struct texture texture;

typedef color (*texture_value_fn)(texture* self, double u, double v, point3 p);

struct texture{
  texture_value_fn value;
  void* data;
};



#endif
