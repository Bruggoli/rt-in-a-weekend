#ifndef PERLIN_H
#define PERLIN_H
#include "../core/vec3.h"

#define POINT_COUNT 256

// The book wants me to initialise the size of the array in the struct
// Its not doable like they do in the book, so instead of declaring
// and initialising point_count in the struct, we define it up top
typedef struct{
  // initialise to the size of POINT_COUNT
  double randfloat[POINT_COUNT];
  int perm_x[POINT_COUNT],
      perm_y[POINT_COUNT],
      perm_z[POINT_COUNT];
} perlin;

perlin* perlin_create();

double noise_create(perlin* per, point3 p);

void render_noise_map(char* filename);

#endif
