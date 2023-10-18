#include <gtest/gtest.h>

#include "simdvec.h"
#include "test_fixture.hpp"

TEST_F(VectorFixture, TestDotProduct) {
  float expected = dot_product();
  float actual = ::dot_product(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestDotProductSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = dot_product();
  float actual = ::dot_product_sse(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestDotProductAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = dot_product();
  float actual = ::dot_product_avx(x.size(), x.data(), y.data());
  EXPECT_NEAR(expected, actual, 1e-6);
}

TEST_F(VectorFixture, TestDotProductAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = dot_product();
  float actual = ::dot_product_avx512f(x.size(), x.data(), y.data());
  EXPECT_NEAR(expected, actual, 1e-6);
}
