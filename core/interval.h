#ifndef INTERVAL_H
#define INTERVAL_H

#include <stdbool.h>
#include <math.h>

typedef struct {
  double min, max;
} interval;
interval interval_create(double min, double max);

interval interval_empty(void);

double interval_size(interval i);

interval interval_enclose(interval a, interval b);

bool interval_contains(interval i, double x);

bool interval_surrounds(interval i, double x);

double clamp(interval i, double x);

interval interval_expand(interval i, double delta);

#endif
