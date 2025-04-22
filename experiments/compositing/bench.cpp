#include <benchmark/benchmark.h>

#include "impl.h"
#include "utils.h"

void BM_Empty(benchmark::State& state) {
  int a = 0;
  int b = 0;
  int c = 0;
  for (auto _ : state) {
    a = add(1, c);
    b = mul(3, a);
    c = add(a, b);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BM_Empty);
BENCHMARK_MAIN();