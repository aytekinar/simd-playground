#ifndef __SIMDVEC_ATTRIBUTE_H__
#define __SIMDVEC_ATTRIBUTE_H__

#ifdef __cplusplus
extern "C" {
#define restrict __restrict
#endif

#include <stdint.h>

float l2_distance(const int16_t, const float *restrict, const float *restrict);
float inner_product(const int16_t, const float *restrict,
                    const float *restrict);
float cosine_distance(const int16_t, const float *restrict,
                      const float *restrict);
float l1_distance(const int16_t, const float *restrict, const float *restrict);
float l2_norm(const int16_t, const float *);

#ifdef __cplusplus
}
#endif

#endif // __SIMDVEC_ATTRIBUTE_H__
