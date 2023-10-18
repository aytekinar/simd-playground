#ifndef _DISTANCE_H_
#define _DISTANCE_H_

#ifdef __cplusplus
extern "C" {
#define restrict __restrict
#endif

#include <stddef.h>

int is_sse_supported(void);
int is_avx_supported(void);
int is_avx512f_supported(void);

void add(const size_t n, const float *restrict x, const float *restrict y,
         float *restrict z);
void add_sse(const size_t n, const float *restrict x, const float *restrict y,
             float *restrict z);
void add_avx(const size_t n, const float *restrict x, const float *restrict y,
             float *restrict z);
void add_avx512f(const size_t n, const float *restrict x,
                 const float *restrict y, float *restrict z);

void sub(const size_t n, const float *restrict x, const float *restrict y,
         float *restrict z);
void sub_sse(const size_t n, const float *restrict x, const float *restrict y,
             float *restrict z);
void sub_avx(const size_t n, const float *restrict x, const float *restrict y,
             float *restrict z);
void sub_avx512f(const size_t n, const float *restrict x,
                 const float *restrict y, float *restrict z);

float dot_product(const size_t n, const float *restrict x,
                  const float *restrict y);
float dot_product_sse(const size_t n, const float *restrict x,
                      const float *restrict y);
float dot_product_avx(const size_t n, const float *restrict x,
                      const float *restrict y);
float dot_product_avx512f(const size_t n, const float *restrict x,
                          const float *restrict y);

float cosine_similarity(const size_t n, const float *restrict x,
                        const float *restrict y);
float cosine_similarity_sse(const size_t n, const float *restrict x,
                            const float *restrict y);
float cosine_similarity_avx(const size_t n, const float *restrict x,
                            const float *restrict y);
float cosine_similarity_avx512f(const size_t n, const float *restrict x,
                                const float *restrict y);

float l1_distance(const size_t n, const float *restrict x,
                  const float *restrict y);
float l1_distance_sse(const size_t n, const float *restrict x,
                      const float *restrict y);
float l1_distance_avx(const size_t n, const float *restrict x,
                      const float *restrict y);
float l1_distance_avx512f(const size_t n, const float *restrict x,
                          const float *restrict y);

float l1_norm(const size_t n, const float *x);
float l1_norm_sse(const size_t n, const float *x);
float l1_norm_avx(const size_t n, const float *x);
float l1_norm_avx512f(const size_t n, const float *x);

float l2_distance(const size_t n, const float *restrict x,
                  const float *restrict y);
float l2_distance_sse(const size_t n, const float *restrict x,
                      const float *restrict y);
float l2_distance_avx(const size_t n, const float *restrict x,
                      const float *restrict y);
float l2_distance_avx512f(const size_t n, const float *restrict x,
                          const float *restrict y);

float l2_norm(const size_t n, const float *x);
float l2_norm_sse(const size_t n, const float *x);
float l2_norm_avx(const size_t n, const float *x);
float l2_norm_avx512f(const size_t n, const float *x);

#ifdef __cplusplus
}
#endif

#endif // _DISTANCE_H_
