#include <gtest/gtest.h>

#include "simdvec.h"
#include "test_fixture.hpp"

TEST_F(VectorFixture, TestL1Distance) {
  float expected = l1_distance();
  float actual = ::l1_distance(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1DistanceSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = l1_distance();
  float actual = ::l1_distance_sse(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1DistanceAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = l1_distance();
  float actual = ::l1_distance_avx(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1DistanceAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = l1_distance();
  float actual = ::l1_distance_avx512f(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1Norm) {
  float expected = l1_norm();
  float actual = ::l1_norm(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1NormSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = l1_norm();
  float actual = ::l1_norm_sse(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1NormAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = l1_norm();
  float actual = ::l1_norm_avx(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestL1NormAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = l1_norm();
  float actual = ::l1_norm_avx512f(x.size(), x.data());
  EXPECT_FLOAT_EQ(expected, actual);
}
