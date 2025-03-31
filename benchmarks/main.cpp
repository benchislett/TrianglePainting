#include <benchmark/benchmark.h>

#include "shaders.h"
#include "utils.h"

#include <random>

float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

RGBA01 random_rgba01() {
    return RGBA01{
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f)
    };
}

RGBA255 random_rgba255() {
    return RGBA255{
        static_cast<unsigned char>(random_int(1, 254)),
        static_cast<unsigned char>(random_int(1, 254)),
        static_cast<unsigned char>(random_int(1, 254)),
        static_cast<unsigned char>(random_int(1, 254))
    };
}

static void BM_CompositOverStraight255(benchmark::State& state) {
    const int kNumPixels = 1024;
    RGBA255 foreground = random_rgba255();
    for (auto _ : state) {
        // Prepare inputs without timing overhead.
        state.PauseTiming();
        RGBA255* background = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        for (int i = 0; i < kNumPixels; ++i) {
            background[i] = random_rgba255();
            background[i].a = 255;
        }
        RGBA255* outputs = (RGBA255*) aligned_alloc(64, sizeof(RGBA255) * kNumPixels);
        state.ResumeTiming();

        // Measured work: composite each pixel.
        for (int i = 0; i < kNumPixels; ++i) {
            RGBA255 result = composit_over_straight_255(background[i], foreground);
            benchmark::DoNotOptimize(result);
            outputs[i] = result;
        }
        benchmark::ClobberMemory();
    }
}

static void BM_CompositOverStraight255_SSE(benchmark::State& state) {
    const int kNumPixels = 1024;
    RGBA255 foreground = random_rgba255();
    for (auto _ : state) {
        // Prepare inputs without timing overhead.
        state.PauseTiming();
        RGBA255* background = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        for (int i = 0; i < kNumPixels; ++i) {
            background[i] = random_rgba255();
            background[i].a = 255;
        }
        RGBA255* outputs = (RGBA255*) aligned_alloc(64, sizeof(RGBA255) * kNumPixels);
        state.ResumeTiming();

        // Measured work: composite each pixel.
        for (int i = 0; i < kNumPixels / 4; ++i) {
            __m128i result = alphaBlendSSE((unsigned char*)(&background[i * 4]), foreground.r, foreground.g, foreground.b, foreground.a);
            benchmark::DoNotOptimize(result);
            *((__m128i*)&outputs[i * 4]) = result;
        }
        benchmark::ClobberMemory();
    }
}

static void BM_CompositOverStraight255_AVX(benchmark::State& state) {
    const int kNumPixels = 1024;
    RGBA255 foreground = random_rgba255();
    for (auto _ : state) {
        // Prepare inputs without timing overhead.
        state.PauseTiming();
        RGBA255* background = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        for (int i = 0; i < kNumPixels; ++i) {
            background[i] = random_rgba255();
            background[i].a = 255;
        }
        RGBA255* outputs = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        state.ResumeTiming();

        // Measured work: composite each pixel.
        for (int i = 0; i < kNumPixels / 8; ++i) {
            __m256i result = alphaBlendAVX((unsigned char*)(&background[i * 8]), foreground.r, foreground.g, foreground.b, foreground.a);
            benchmark::DoNotOptimize(result);
            *((__m256i*)&outputs[i * 8]) = result;
        }
        benchmark::ClobberMemory();
    }
}

static void BM_CompositOverStraight255_AVX_SinglePath(benchmark::State& state) {
    const int kNumPixels = 1024;
    RGBA255 foreground = random_rgba255();
    for (auto _ : state) {
        // Prepare inputs without timing overhead.
        state.PauseTiming();
        RGBA255* background = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        for (int i = 0; i < kNumPixels; ++i) {
            background[i] = random_rgba255();
            background[i].a = 255;
        }
        RGBA255* outputs = (RGBA255*) aligned_alloc(128, sizeof(RGBA255) * kNumPixels);
        state.ResumeTiming();

        // Measured work: composite each pixel.
        for (int i = 0; i < kNumPixels / 8; ++i) {
            __m256i result = alphaBlendAVX2_SinglePath((unsigned char*)(&background[i * 8]), foreground.r, foreground.g, foreground.b, foreground.a);
            benchmark::DoNotOptimize(result);
            *((__m256i*)&outputs[i * 8]) = result;
        }
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_CompositOverStraight255);
BENCHMARK(BM_CompositOverStraight255_SSE);
BENCHMARK(BM_CompositOverStraight255_AVX);
BENCHMARK(BM_CompositOverStraight255_AVX_SinglePath);

BENCHMARK_MAIN();
