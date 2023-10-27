#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "simdvec-attribute.h"

static void BM_dot_product(benchmark::State &state) {
  const int16_t dim = state.range(0);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(inner_product(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_dot_product)
    ->ArgsProduct({
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_cosine_distance(benchmark::State &state) {
  const int16_t dim = state.range(0);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(cosine_distance(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_cosine_distance)
    ->ArgsProduct({
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l1_distance(benchmark::State &state) {
  const int16_t dim = state.range(0);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(l1_distance(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l1_distance)
    ->ArgsProduct({
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l2_distance(benchmark::State &state) {
  const int16_t dim = state.range(0);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim), y(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
  std::generate(y.begin(), y.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(l2_distance(dim, x.data(), y.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l2_distance)
    ->ArgsProduct({
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

static void BM_l2_norm(benchmark::State &state) {
  const int16_t dim = state.range(0);
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(dim);
  std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    benchmark::DoNotOptimize(l2_norm(dim, x.data()));
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_l2_norm)
    ->ArgsProduct({
        benchmark::CreateRange(1 << 10, 16 * (1 << 10), 2),
    });

BENCHMARK_MAIN();
