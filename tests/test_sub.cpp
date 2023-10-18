#include <gtest/gtest.h>

#include "simdvec.h"
#include "test_fixture.hpp"

TEST_F(VectorFixture, TestSub) {
  std::vector<float> expected = sub();
  std::vector<float> actual(x.size());
  ::sub(x.size(), x.data(), y.data(), actual.data());
  for (size_t idx = 0; idx < x.size(); ++idx) {
    EXPECT_FLOAT_EQ(expected[idx], actual[idx]);
  }
}

TEST_F(VectorFixture, TestSubSSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  std::vector<float> expected = sub();
  std::vector<float> actual(x.size());
  ::sub_sse(x.size(), x.data(), y.data(), actual.data());
  for (size_t idx = 0; idx < x.size(); ++idx) {
    EXPECT_FLOAT_EQ(expected[idx], actual[idx]);
  }
}

TEST_F(VectorFixture, TestSubAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  std::vector<float> expected = sub();
  std::vector<float> actual(x.size());
  ::sub_avx(x.size(), x.data(), y.data(), actual.data());
  for (size_t idx = 0; idx < x.size(); ++idx) {
    EXPECT_FLOAT_EQ(expected[idx], actual[idx]);
  }
}

TEST_F(VectorFixture, TestSubAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  std::vector<float> expected = sub();
  std::vector<float> actual(x.size());
  ::sub_avx512f(x.size(), x.data(), y.data(), actual.data());
  for (size_t idx = 0; idx < x.size(); ++idx) {
    EXPECT_FLOAT_EQ(expected[idx], actual[idx]);
  }
}
