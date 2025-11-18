#include "noise_texture.h"
#include "perlin.h"
#include <math.h>
#include <stdlib.h>

texture* noise_texture_create(double scale) {
  noise_texture* nt = malloc(sizeof(noise_texture));
  texture* tx = malloc(sizeof(texture));
  perlin* p = perlin_create();
  nt->noise = *p;
  nt->scale = scale;
  free(p);
  tx->data = nt;
  tx->value = noise_texture_value;
  return tx;
}

color noise_texture_value(texture* self, double u, double v, point3 p) {
  noise_texture* nt = self->data;
  double noise_val = turb_create(&nt->noise, vec3_scale(p, nt->scale), 7);
  double phased_noise_val = 1 + sin(nt->scale * p.e[2] + 10 * noise_val);
  vec3 color = vec3_scale(color_create(0.5, 0.5, 0.5), phased_noise_val);
  return color;
}


