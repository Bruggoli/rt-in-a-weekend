#ifndef PERLIN_H
#define PERLIN_H
#include "../core/vec3.h"

#define POINT_COUNT 256

// The book wants me to initialise the size of the array in the struct
// Its not doable like they do in the book, so instead of declaring
// and initialising point_count in the struct, we define it up top
typedef struct{
  // initialise to the size of POINT_COUNT
  vec3 randvec[POINT_COUNT];
  int perm_x[POINT_COUNT],
      perm_y[POINT_COUNT],
      perm_z[POINT_COUNT];
} perlin;

perlin* perlin_create();
static inline double perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
  double uu = u*u*(3-2*u);
  double vv = v*v*(3-2*v);
  double ww = w*w*(3-2*w);
  double accum = 0.0;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        vec3 weight_v = vec3_create(u-i, v-j, w-k);
        accum += (i*uu + (1-i)*(1-uu))
                 * (j*vv + (1-j)*(1-vv))
                 * (k*ww + (1-k)*(1-ww))
                 * vec3_dot(c[i][j][k], weight_v);
      }
    }
  }
  return accum;
}

double noise_create(perlin* per, point3 p);


double turb_create(perlin* per, point3 p, int depth);

void render_noise_map(char* filename);

#endif
