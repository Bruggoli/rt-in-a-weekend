#include "noise_texture.h"
#include "perlin.h"
#include <stdlib.h>

texture* noise_texture_create() {
  noise_texture* nt = malloc(sizeof(noise_texture));
  texture* tx = malloc(sizeof(texture));
  perlin* p = perlin_create();
  nt->noise = *p;
  free(p);
  tx->data = nt;
  tx->value = noise_texture_value;
  return tx;
}

color noise_texture_value(texture* self, double u, double v, point3 p) {
  noise_texture* nt = self->data;
  double noise_val = noise_create(&nt->noise, p);
  static int count = 0;
  if (count < 10) {
    fprintf(stderr, "Noise #%d: %.3f\n", count++, noise_val);
  }
  return vec3_scale(color_create(1, 1, 1), noise_val);
}


