#include <gtest/gtest.h>

#include "simdvec.h"
#include "test_fixture.hpp"

TEST_F(VectorFixture, TestCosineSimilarity) {
  float expected = cosine_similarity();
  float actual = ::cosine_similarity(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestCosineSimilaritySSE) {
  if (!is_sse_supported()) {
    GTEST_SKIP() << "SSE is not supported on this machine.";
  }
  float expected = cosine_similarity();
  float actual = ::cosine_similarity_sse(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestCosineSimilarityAVX) {
  if (!is_avx_supported()) {
    GTEST_SKIP() << "AVX is not supported on this machine.";
  }
  float expected = cosine_similarity();
  float actual = ::cosine_similarity_avx(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}

TEST_F(VectorFixture, TestCosineSimilarityAVX512F) {
  if (!is_avx512f_supported()) {
    GTEST_SKIP() << "AVX512F is not supported on this machine.";
  }
  float expected = cosine_similarity();
  float actual = ::cosine_similarity_avx512f(x.size(), x.data(), y.data());
  EXPECT_FLOAT_EQ(expected, actual);
}
