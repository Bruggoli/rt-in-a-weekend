#include <stdio.h>
#include <stdlib.h>
#include "perlin.h"
#include "../utils/rtweekend.h"


static void perlin_generate_perm(int* p);
static void permute(int* p, int n);

perlin* perlin_create() {
  fprintf(stderr, "Initializing perlin...\n");
  perlin* p = malloc(sizeof(perlin));

  for (int i = 0; i < POINT_COUNT; i++) {
    p->randfloat[i] = random_double();
  }

  perlin_generate_perm(p->perm_x);
  perlin_generate_perm(p->perm_y);
  perlin_generate_perm(p->perm_z);
  fprintf(stderr, "Perlin initialized!\n");
  return p;
}

double noise_create(perlin* per, point3 p) {
  double scale = 4.0;
  int i = (int)(scale*p.e[0]) & 255;
  int j = (int)(scale*p.e[1]) & 255;
  int k = (int)(scale*p.e[2]) & 255;

  int index = per->perm_x[i] ^ per->perm_y[j] ^ per->perm_z[k];
  static int count = 0;
  if (count < 10) {
    fprintf(stderr, "perm_x[%d]=%d, perm_y[%d]=%d, perm_z[%d]=%d -> XOR=%d\n",
            i, per->perm_x[i], j, per->perm_y[j], k, per->perm_z[k],
            per->perm_x[i] ^ per->perm_y[j] ^ per->perm_z[k]);
    count++;
  }  

  return per->randfloat[index];
}

static void perlin_generate_perm(int* p) {
  for (int i = 0; i < POINT_COUNT; i++) {
    p[i] = i;
  }

  permute(p, POINT_COUNT);
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
