#include <stdio.h>
#include <stdlib.h>
#include "perlin.h"
#include "../utils/rtweekend.h"


static void perlin_generate_perm(int* p);
static void permute(int* p, int n);
static double perlin_interp(vec3 c[2][2][2], double u, double v, double w);

perlin* perlin_create() {
  fprintf(stderr, "Initializing perlin...\n");
  perlin* p = malloc(sizeof(perlin));

  for (int i = 0; i < POINT_COUNT; i++) {
    p->randvec[i] = unit_vector(vec3_random_range(-1, 1));
  }

  perlin_generate_perm(p->perm_x);
  perlin_generate_perm(p->perm_y);
  perlin_generate_perm(p->perm_z);
  fprintf(stderr, "Perlin initialized!\n");
  return p;
}


double turb_create(perlin* per, point3 p, int depth) {
  double accum = 0.0;
  point3 temp_p = p;
  double weight = 1.0;

  for (int i = 0; i < depth; i++) {
    accum += weight * noise_create(per, temp_p);
    weight *= 0.5;
    temp_p = vec3_scale(temp_p, 2);
  }

  return fabs(accum);
}



static void perlin_generate_perm(int* p) {
  for (int i = 0; i < POINT_COUNT; i++) {
    p[i] = i;
  }

  permute(p, POINT_COUNT);
}

double noise_create(perlin* per, point3 p) {
  double u = p.e[0] - floor(p.e[0]);
  double v = p.e[1] - floor(p.e[1]);
  double w = p.e[2] - floor(p.e[2]);


  int i = (int)floor(p.e[0]);
  int j = (int)floor(p.e[1]);
  int k = (int)floor(p.e[2]);
  vec3 c[2][2][2];

  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      for (int dk = 0; dk < 2; dk++) {
        c[di][dj][dk] = per->randvec[
          per->perm_x[(i+di) & 255] ^
          per->perm_y[(j+dj) & 255] ^
          per->perm_z[(k+dk) & 255]
        ];
      }
    }
  }

  return perlin_interp(c, u, v, w);
}

static void permute(int* p, int n) {
  for (int i = n-1; i > 0; i--) {
    int target = random_int(0, i);
    int tmp = p[i];
    p[i] = p[target];
    p[target] = tmp;
  }
}

void render_noise_map(char* filename) {
  FILE* f = fopen(filename, "w");
  int width = POINT_COUNT;
  int height = POINT_COUNT;

  perlin* p = perlin_create();

  fprintf(f, "P3\n%d %d\n255\n", POINT_COUNT, POINT_COUNT);

  for (int j = 0; j < POINT_COUNT; j++) {
    fprintf(stderr, "\rScanlines remaining: %d/%d", (POINT_COUNT - j), POINT_COUNT);
    fflush(stderr);
    for (int i = 0; i < POINT_COUNT; i++) {
      point3 point = vec3_create(i, j, 0);
      double noise_val = noise_create(p, point);
      int pixel_color = (int)(255 * noise_val);
      fprintf(f, "%d %d %d\n", pixel_color, pixel_color, pixel_color);
      }
    
  }
  fclose(f);

}
