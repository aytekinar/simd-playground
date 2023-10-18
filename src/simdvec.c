#include <immintrin.h>
#include <math.h>

#include "simdvec.h"

#ifdef _MSC_VER
#include <intrin.h>
#define __attribute__(x)
#define __aligned__(x) __declspec(align(x))
#elif defined(__GNUC__) || defined(__clang__)
#define __aligned__(x) __attribute__((aligned(x)))
#endif

int is_sse_supported(void) {
#ifdef _MSC_VER
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int nIds = cpuInfo[0];

  if (nIds >= 1) {
    __cpuidex(cpuInfo, 1, 0);
    return (cpuInfo[3] & (1 << 25)) != 0;
  }

  return 0;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("sse");
#else
  return 0;
#endif
}

int is_avx_supported(void) {
#ifdef _MSC_VER
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int nIds = cpuInfo[0];

  if (nIds >= 1) {
    __cpuidex(cpuInfo, 1, 0);
    return (cpuInfo[2] & (1 << 28)) != 0;
  }

  return 0;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx");
#else
  return 0;
#endif
}

int is_avx512f_supported(void) {
#ifdef _MSC_VER
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int nIds = cpuInfo[0];

  if (nIds >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    return (cpuInfo[1] & (1 << 16)) != 0;
  }

  return 0;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx512f");
#else
  return 0;
#endif
}

__attribute__((target("default"))) void add(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  for (size_t idx = 0; idx < n; idx++) {
    z[idx] = x[idx] + y[idx];
  }
}

__attribute__((target("sse"))) void add_sse(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1, v2, sum;
  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    sum = _mm_add_ps(v1, v2);
    _mm_storeu_ps(z + idx, sum);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] + y[idx];
  }
}

__attribute__((target("avx"))) void add_avx(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1, v2, sum;
  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    sum = _mm256_add_ps(v1, v2);
    _mm256_storeu_ps(z + idx, sum);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] + y[idx];
  }

  _mm256_zeroupper();
}

__attribute__((target("avx512f"))) void add_avx512f(const size_t n,
                                                    const float *restrict x,
                                                    const float *restrict y,
                                                    float *restrict z) {
  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1, v2, sum;
  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    sum = _mm512_add_ps(v1, v2);
    _mm512_storeu_ps(z + idx, sum);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] + y[idx];
  }

  _mm256_zeroupper();
}

__attribute__((target("default"))) void sub(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  for (size_t idx = 0; idx < n; idx++) {
    z[idx] = x[idx] - y[idx];
  }
}

__attribute__((target("sse"))) void sub_sse(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1, v2, diff;
  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    diff = _mm_sub_ps(v1, v2);
    _mm_storeu_ps(z + idx, diff);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] - y[idx];
  }
}

__attribute__((target("avx"))) void sub_avx(const size_t n,
                                            const float *restrict x,
                                            const float *restrict y,
                                            float *restrict z) {
  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1, v2, diff;
  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    diff = _mm256_sub_ps(v1, v2);
    _mm256_storeu_ps(z + idx, diff);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] - y[idx];
  }

  _mm256_zeroupper();
}

__attribute__((target("avx512f"))) void sub_avx512f(const size_t n,
                                                    const float *restrict x,
                                                    const float *restrict y,
                                                    float *restrict z) {
  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1, v2, diff;
  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    diff = _mm512_sub_ps(v1, v2);
    _mm512_storeu_ps(z + idx, diff);
  }

  for (/* no init */; idx < n; idx++) {
    z[idx] = x[idx] - y[idx];
  }

  _mm256_zeroupper();
}

__attribute__((target("default"))) float
dot_product(const size_t n, const float *restrict x, const float *restrict y) {
  float sum = 0;
  for (size_t idx = 0; idx < n; idx++) {
    sum += x[idx] * y[idx];
  }
  return sum;
}

__attribute__((target("sse"))) float dot_product_sse(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(16) tmp[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1, v2;
  __m128 sum = _mm_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
  }
  _mm_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * y[idx];
  }

  return res;
}

__attribute__((target("avx"))) float dot_product_avx(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(32) tmp[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1, v2;
  __m256 sum = _mm256_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
  }
  _mm256_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * y[idx];
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("avx512f"))) float
dot_product_avx512f(const size_t n, const float *restrict x,
                    const float *restrict y) {
  float __aligned__(64) tmp[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1, v2;
  __m512 sum = _mm512_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    sum = _mm512_fmadd_ps(v1, v2, sum);
  }
  _mm512_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
        tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] +
        tmp[15];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * y[idx];
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("default"))) float
cosine_similarity(const size_t n, const float *restrict x,
                  const float *restrict y) {
  float dot = 0;
  float norm_x = 0;
  float norm_y = 0;
  for (size_t idx = 0; idx < n; idx++) {
    dot += x[idx] * y[idx];
    norm_x += x[idx] * x[idx];
    norm_y += y[idx] * y[idx];
  }
  return dot / sqrtf(norm_x * norm_y);
}

__attribute__((target("sse"))) float
cosine_similarity_sse(const size_t n, const float *restrict x,
                      const float *restrict y) {
  float __aligned__(16) tmp_sum[sizeof(__m128) / sizeof(float)];
  float __aligned__(16) tmp_nx[sizeof(__m128) / sizeof(float)];
  float __aligned__(16) tmp_ny[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1, v2;
  __m128 sum = _mm_setzero_ps();
  __m128 nx = _mm_setzero_ps();
  __m128 ny = _mm_setzero_ps();
  float res_sum, res_nx, res_ny;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
    nx = _mm_add_ps(nx, _mm_mul_ps(v1, v1));
    ny = _mm_add_ps(ny, _mm_mul_ps(v2, v2));
  }
  _mm_store_ps(tmp_sum, sum);
  _mm_store_ps(tmp_nx, nx);
  _mm_store_ps(tmp_ny, ny);
  res_sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3];
  res_nx = tmp_nx[0] + tmp_nx[1] + tmp_nx[2] + tmp_nx[3];
  res_ny = tmp_ny[0] + tmp_ny[1] + tmp_ny[2] + tmp_ny[3];

  for (/* no init */; idx < n; idx++) {
    res_sum += x[idx] * y[idx];
    res_nx += x[idx] * x[idx];
    res_ny += y[idx] * y[idx];
  }

  return res_sum / sqrtf(res_nx * res_ny);
}

__attribute__((target("avx"))) float
cosine_similarity_avx(const size_t n, const float *restrict x,
                      const float *restrict y) {
  float __aligned__(32) tmp_sum[sizeof(__m256) / sizeof(float)];
  float __aligned__(32) tmp_nx[sizeof(__m256) / sizeof(float)];
  float __aligned__(32) tmp_ny[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1, v2;
  __m256 sum = _mm256_setzero_ps();
  __m256 nx = _mm256_setzero_ps();
  __m256 ny = _mm256_setzero_ps();
  float res_sum, res_nx, res_ny;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    nx = _mm256_add_ps(nx, _mm256_mul_ps(v1, v1));
    ny = _mm256_add_ps(ny, _mm256_mul_ps(v2, v2));
  }
  _mm256_store_ps(tmp_sum, sum);
  _mm256_store_ps(tmp_nx, nx);
  _mm256_store_ps(tmp_ny, ny);
  res_sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3] + tmp_sum[4] +
            tmp_sum[5] + tmp_sum[6] + tmp_sum[7];
  res_nx = tmp_nx[0] + tmp_nx[1] + tmp_nx[2] + tmp_nx[3] + tmp_nx[4] +
           tmp_nx[5] + tmp_nx[6] + tmp_nx[7];
  res_ny = tmp_ny[0] + tmp_ny[1] + tmp_ny[2] + tmp_ny[3] + tmp_ny[4] +
           tmp_ny[5] + tmp_ny[6] + tmp_ny[7];

  for (/* no init */; idx < n; idx++) {
    res_sum += x[idx] * y[idx];
    res_nx += x[idx] * x[idx];
    res_ny += y[idx] * y[idx];
  }

  _mm256_zeroupper();

  return res_sum / sqrtf(res_nx * res_ny);
}

__attribute__((target("avx512f"))) float
cosine_similarity_avx512f(const size_t n, const float *restrict x,
                          const float *restrict y) {
  float __aligned__(64) tmp_sum[sizeof(__m512) / sizeof(float)];
  float __aligned__(64) tmp_nx[sizeof(__m512) / sizeof(float)];
  float __aligned__(64) tmp_ny[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1, v2;
  __m512 sum = _mm512_setzero_ps();
  __m512 nx = _mm512_setzero_ps();
  __m512 ny = _mm512_setzero_ps();
  float res_sum, res_nx, res_ny;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    sum = _mm512_fmadd_ps(v1, v2, sum);
    nx = _mm512_fmadd_ps(v1, v1, nx);
    ny = _mm512_fmadd_ps(v2, v2, ny);
  }
  _mm512_store_ps(tmp_sum, sum);
  _mm512_store_ps(tmp_nx, nx);
  _mm512_store_ps(tmp_ny, ny);
  res_sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3] + tmp_sum[4] +
            tmp_sum[5] + tmp_sum[6] + tmp_sum[7] + tmp_sum[8] + tmp_sum[9] +
            tmp_sum[10] + tmp_sum[11] + tmp_sum[12] + tmp_sum[13] +
            tmp_sum[14] + tmp_sum[15];
  res_nx = tmp_nx[0] + tmp_nx[1] + tmp_nx[2] + tmp_nx[3] + tmp_nx[4] +
           tmp_nx[5] + tmp_nx[6] + tmp_nx[7] + tmp_nx[8] + tmp_nx[9] +
           tmp_nx[10] + tmp_nx[11] + tmp_nx[12] + tmp_nx[13] + tmp_nx[14] +
           tmp_nx[15];
  res_ny = tmp_ny[0] + tmp_ny[1] + tmp_ny[2] + tmp_ny[3] + tmp_ny[4] +
           tmp_ny[5] + tmp_ny[6] + tmp_ny[7] + tmp_ny[8] + tmp_ny[9] +
           tmp_ny[10] + tmp_ny[11] + tmp_ny[12] + tmp_ny[13] + tmp_ny[14] +
           tmp_ny[15];

  for (/* no init */; idx < n; idx++) {
    res_sum += x[idx] * y[idx];
    res_nx += x[idx] * x[idx];
    res_ny += y[idx] * y[idx];
  }

  _mm256_zeroupper();

  return res_sum / sqrtf(res_nx * res_ny);
}

__attribute__((target("default"))) float
l1_distance(const size_t n, const float *restrict x, const float *restrict y) {
  float sum = 0;
  for (size_t idx = 0; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    sum += fabsf(diff);
  }
  return sum;
}

__attribute__((target("sse"))) float l1_distance_sse(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(16) tmp[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 diff, v1, v2;
  __m128 sum = _mm_setzero_ps();
  __m128 sign_bit = _mm_set1_ps(-0.0f);
  float res;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_andnot_ps(sign_bit, diff));
  }
  _mm_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += fabsf(diff);
  }

  return res;
}

__attribute__((target("avx"))) float l1_distance_avx(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(32) tmp[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 diff, v1, v2;
  __m256 sum = _mm256_setzero_ps();
  __m256 sign_bit = _mm256_set1_ps(-0.0f);
  float res;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_andnot_ps(sign_bit, diff));
  }
  _mm256_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += fabsf(diff);
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("avx512f"))) float
l1_distance_avx512f(const size_t n, const float *restrict x,
                    const float *restrict y) {
  float __aligned__(64) tmp[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 diff, v1, v2;
  __m512 sum = _mm512_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_add_ps(sum, _mm512_abs_ps(diff));
  }
  _mm512_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
        tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] +
        tmp[15];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += fabsf(diff);
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("default"))) float l1_norm(const size_t n,
                                                 const float *x) {
  float sum = 0;
  for (size_t idx = 0; idx < n; idx++) {
    sum += fabsf(x[idx]);
  }
  return sum;
}

__attribute__((target("sse"))) float l1_norm_sse(const size_t n,
                                                 const float *x) {
  float __aligned__(16) tmp[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1;
  __m128 sum = _mm_setzero_ps();
  __m128 sign_bit = _mm_set1_ps(-0.0f);
  float res;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    sum = _mm_add_ps(sum, _mm_andnot_ps(sign_bit, v1));
  }
  _mm_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (/* no init */; idx < n; idx++) {
    res += fabsf(x[idx]);
  }

  return res;
}

__attribute__((target("avx"))) float l1_norm_avx(const size_t n,
                                                 const float *x) {
  float __aligned__(32) tmp[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1;
  __m256 sum = _mm256_setzero_ps();
  __m256 sign_bit = _mm256_set1_ps(-0.0f);
  float res;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    sum = _mm256_add_ps(sum, _mm256_andnot_ps(sign_bit, v1));
  }
  _mm256_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (/* no init */; idx < n; idx++) {
    res += fabsf(x[idx]);
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("avx512f"))) float l1_norm_avx512f(const size_t n,
                                                         const float *x) {
  float __aligned__(64) tmp[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1;
  __m512 sum = _mm512_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    sum = _mm512_add_ps(sum, _mm512_abs_ps(v1));
  }
  _mm512_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
        tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] +
        tmp[15];

  for (/* no init */; idx < n; idx++) {
    res += fabsf(x[idx]);
  }

  _mm256_zeroupper();

  return res;
}

__attribute__((target("default"))) float
l2_distance(const size_t n, const float *restrict x, const float *restrict y) {
  float sum = 0;
  for (size_t idx = 0; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

__attribute__((target("sse"))) float l2_distance_sse(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(16) tmp[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 diff, v1, v2;
  __m128 sum = _mm_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    v2 = _mm_loadu_ps(y + idx);
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }
  _mm_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += diff * diff;
  }

  return sqrtf(res);
}

__attribute__((target("avx"))) float l2_distance_avx(const size_t n,
                                                     const float *restrict x,
                                                     const float *restrict y) {
  float __aligned__(32) tmp[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 diff, v1, v2;
  __m256 sum = _mm256_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    v2 = _mm256_loadu_ps(y + idx);
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }
  _mm256_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += diff * diff;
  }

  _mm256_zeroupper();

  return sqrtf(res);
}

__attribute__((target("avx512f"))) float
l2_distance_avx512f(const size_t n, const float *restrict x,
                    const float *restrict y) {
  float __aligned__(64) tmp[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 diff, v1, v2;
  __m512 sum = _mm512_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    v2 = _mm512_loadu_ps(y + idx);
    diff = _mm512_sub_ps(v1, v2);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }
  _mm512_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
        tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] +
        tmp[15];

  for (/* no init */; idx < n; idx++) {
    const float diff = x[idx] - y[idx];
    res += diff * diff;
  }

  _mm256_zeroupper();

  return sqrtf(res);
}

__attribute__((target("default"))) float l2_norm(const size_t n,
                                                 const float *x) {
  float sum = 0;
  for (size_t idx = 0; idx < n; idx++) {
    sum += x[idx] * x[idx];
  }
  return sqrtf(sum);
}

__attribute__((target("sse"))) float l2_norm_sse(const size_t n,
                                                 const float *x) {
  float __aligned__(16) tmp[sizeof(__m128) / sizeof(float)];

  const size_t nbatch = (n / 4) * 4;
  size_t idx = 0;

  __m128 v1;
  __m128 sum = _mm_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 4) {
    v1 = _mm_loadu_ps(x + idx);
    sum = _mm_add_ps(sum, _mm_mul_ps(v1, v1));
  }
  _mm_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * x[idx];
  }

  return sqrtf(res);
}

__attribute__((target("avx"))) float l2_norm_avx(const size_t n,
                                                 const float *x) {
  float __aligned__(32) tmp[sizeof(__m256) / sizeof(float)];

  const size_t nbatch = (n / 8) * 8;
  size_t idx = 0;

  __m256 v1;
  __m256 sum = _mm256_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 8) {
    v1 = _mm256_loadu_ps(x + idx);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v1));
  }
  _mm256_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * x[idx];
  }

  _mm256_zeroupper();

  return sqrtf(res);
}

__attribute__((target("avx512f"))) float l2_norm_avx512f(const size_t n,
                                                         const float *x) {
  float __aligned__(64) tmp[sizeof(__m512) / sizeof(float)];

  const size_t nbatch = (n / 16) * 16;
  size_t idx = 0;

  __m512 v1;
  __m512 sum = _mm512_setzero_ps();
  float res;

  for (/* no init */; idx < nbatch; idx += 16) {
    v1 = _mm512_loadu_ps(x + idx);
    sum = _mm512_fmadd_ps(v1, v1, sum);
  }
  _mm512_store_ps(tmp, sum);
  res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
        tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] +
        tmp[15];

  for (/* no init */; idx < n; idx++) {
    res += x[idx] * x[idx];
  }

  _mm256_zeroupper();

  return sqrtf(res);
}
