#include <gtest/gtest.h>

#include "simdvec.h"
#include "test_fixture.hpp"

TEST_F(VectorFixture, TestL2Distance) {
  float expected = l2_distance();
  float actual = ::l2_distance(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2DistanceSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = l2_distance();
  float actual = ::l2_distance_sse(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2DistanceAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = l2_distance();
  float actual = ::l2_distance_avx(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2DistanceAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = l2_distance();
  float actual = ::l2_distance_avx512f(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2Norm) {
  float expected = l2_norm();
  float actual = ::l2_norm(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2NormSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = l2_norm();
  float actual = ::l2_norm_sse(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2NormAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = l2_norm();
  float actual = ::l2_norm_avx(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL2NormAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = l2_norm();
  float actual = ::l2_norm_avx512f(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}
