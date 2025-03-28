#include <benchmark/benchmark.h>
#include "shaders.h"

static void BM_OptimalColourShader(benchmark::State& state) {
    // Initialize any required data here
    for (auto _ : state) {
        // Call functions from shaders.h to measure performance
    }
}

BENCHMARK(BM_OptimalColourShader);

BENCHMARK_MAIN();
