#ifndef __TEST_FIXTURE_HPP__
#define __TEST_FIXTURE_HPP__

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

class VectorFixture : public ::testing::Test {
protected:
  VectorFixture() : x(100), y(100) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
    std::generate(y.begin(), y.end(), [&]() { return dist(gen); });
  }

  std::vector<float> add() const {
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), y.begin(), result.begin(),
                   std::plus<float>());
    return result;
  }

  float cosine_similarity() const {
    float dot = std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);
    float mag_x =
        std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0f));
    float mag_y =
        std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0f));
    return dot / (mag_x * mag_y);
  }

  float dot_product() const {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);
  }

  float l1_distance() const {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0f,
                              std::plus<float>(),
                              [](float a, float b) { return std::abs(a - b); });
  }

  float l1_norm() const {
    return std::accumulate(x.begin(), x.end(), 0.0f,
                           [](float a, float b) { return a + std::abs(b); });
  }

  float l2_distance() const {
    return std::sqrt(std::inner_product(
        x.begin(), x.end(), y.begin(), 0.0f, std::plus<float>(),
        [](float a, float b) { return std::pow(a - b, 2); }));
  }

  float l2_norm() const {
    return std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0f));
  }

  std::vector<float> sub() const {
    std::vector<float> result(x.size());
    std::transform(x.begin(), x.end(), y.begin(), result.begin(),
                   std::minus<float>());
    return result;
  }

  virtual ~VectorFixture() = default;

  std::vector<float> x;
  std::vector<float> y;
};

#endif // __TEST_FIXTURE_HPP__
