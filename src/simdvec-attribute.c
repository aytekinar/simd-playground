#include <math.h>
#include <stdint.h>

#include "simdvec-attribute.h"

#if defined NO_ATTRIBUTE  /* no attribute */
#define __attribute__(x)
#elif defined _MSC_VER /* MSVC only */
#define __attribute__(x)
#elif defined __APPLE__ /* Apple/OSX only */
#define target_clones(...)
#elif !defined __has_attribute || !__has_attribute(target_clones) || !defined __gnu_linux__ /* target_clones not supported */
#define target_clones(...)
#endif

__attribute__((target_clones("default", "avx", "fma", "avx512f"))) static float
l2_distance_impl(const int16_t dim, const float *restrict x,
                 const float *restrict y) {
  float distance = 0.0;
  for (int16_t i = 0; i < dim; i++) {
    const float diff = x[i] - y[i];
    distance += diff * diff;
  }
  return distance;
}

__attribute__((target_clones("default", "avx", "fma", "avx512f"))) static float
inner_product_impl(const int16_t dim, const float *restrict x,
                   const float *restrict y) {
  float dot = 0.0;
  for (int16_t i = 0; i < dim; i++)
    dot += x[i] * y[i];
  return dot;
}

__attribute__((target_clones("default", "avx", "fma", "avx512f"))) static float
cosine_distance_impl(const int16_t dim, const float *restrict x,
                     const float *restrict y) {
  float dot = 0.0;
  float normx = 0.0;
  float normy = 0.0;
  for (int16_t i = 0; i < dim; i++) {
    dot += x[i] * y[i];
    normx += x[i] * x[i];
    normy += y[i] * y[i];
  }
  return dot / sqrtf(normx * normy);
}

__attribute__((target_clones("default", "avx", "fma", "avx512f"))) static float
l1_distance_impl(const int16_t dim, const float *restrict x,
                 const float *restrict y) {
  float distance = 0.0;
  for (int16_t i = 0; i < dim; i++)
    distance += fabsf(x[i] - y[i]);
  return distance;
}

__attribute__((target_clones("default", "avx", "fma", "avx512f"))) static float
l2_norm_impl(const int16_t dim, const float *x) {
  float norm = 0.0;
  for (int16_t i = 0; i < dim; i++)
    norm += x[i] * x[i];
  return norm;
}

float l2_distance(const int16_t dim, const float *restrict x,
                  const float *restrict y) {
  return l2_distance_impl(dim, x, y);
}

float inner_product(const int16_t dim, const float *restrict x,
                    const float *restrict y) {
  return inner_product_impl(dim, x, y);
}

float cosine_distance(const int16_t dim, const float *restrict x,
                      const float *restrict y) {
  return cosine_distance_impl(dim, x, y);
}

float l1_distance(const int16_t dim, const float *restrict x,
                  const float *restrict y) {
  return l1_distance_impl(dim, x, y);
}

float l2_norm(const int16_t dim, const float *x) {
  return l2_norm_impl(dim, x);
}
