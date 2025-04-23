#include <benchmark/benchmark.h>

#include "impl.h"
#include "utils.h"

void BM_Empty(benchmark::State& state) {
}

BENCHMARK(BM_Empty);
BENCHMARK_MAIN();