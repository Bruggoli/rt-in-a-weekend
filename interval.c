#include "interval.h"


interval interval_create(double min, double max) {
  return (interval){min, max};
}

interval interval_empty(void) {
  return (interval){INFINITY, -INFINITY};
}

double interval_size(interval i){
  return i.max - i.min;
}

bool interval_contains(interval i, double x) {
  return i.min <= x && x <= i.max;
}

bool interval_surrounds(interval i, double x) {
  return i.min < x && x < i.max;
}

double clamp(interval i, double x) {
  if (x < i.min)
    return i.min;
  if (x > i.max)
    return i.max;
  return x;
}
