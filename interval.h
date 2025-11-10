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

bool interval_contains(interval i, double x);

bool interval_surrounds(interval i, double x);

double clamp(interval i, double x);

#endif
