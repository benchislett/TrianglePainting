#include <benchmark/benchmark.h>

#include "common.h"
#include "pixel_types.h"
#include "composit.h"

/* This file benchmarks scalar compositing operations on different data types. */

template<typename BASETYPE, typename Op>
void BM_composit_scalar_rgba_over_rgba_generic(benchmark::State& state, Op&& op) {
    const int length = state.range(0);
    const int N = length * length;

    using Pixel = RGBA<BASETYPE>;

    Pixel* background = (Pixel*) allocate_aligned(N * sizeof(Pixel));
    Pixel* foreground = (Pixel*) allocate_aligned(N * sizeof(Pixel));

    Pixel* background_initial = (Pixel*) allocate_aligned(N * sizeof(Pixel));
    Pixel* foreground_initial = (Pixel*) allocate_aligned(N * sizeof(Pixel));

    for (int i = 0; i < N; i++) {
        background_initial[i] = random_generic<Pixel>();
        foreground_initial[i] = random_generic<Pixel>();
    }

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(background, background_initial, N * sizeof(Pixel));
        memcpy(foreground, foreground_initial, N * sizeof(Pixel));
        state.ResumeTiming();

        for (int i = 0; i < N; i++) {
            background[i] = op(foreground[i], background[i]);
        }

        benchmark::DoNotOptimize(background);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(uint64_t(state.iterations()));
}

template<typename BASETYPE, typename Op>
void BM_composit_scalar_rgba_over_rgb_generic(benchmark::State& state, Op&& op) {
    const int length = state.range(0);
    const int N = length * length;

    using PixelRGBA = RGBA<BASETYPE>;
    using PixelRGB = RGB<BASETYPE>;

    PixelRGB* background = (PixelRGB*) allocate_aligned(N * sizeof(PixelRGB));
    PixelRGBA* foreground = (PixelRGBA*) allocate_aligned(N * sizeof(PixelRGBA));

    PixelRGB* background_initial = (PixelRGB*) allocate_aligned(N * sizeof(PixelRGB));
    PixelRGBA* foreground_initial = (PixelRGBA*) allocate_aligned(N * sizeof(PixelRGBA));

    for (int i = 0; i < N; i++) {
        background_initial[i] = random_generic<PixelRGB>();
        foreground_initial[i] = random_generic<PixelRGBA>();
    }

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(background, background_initial, N * sizeof(PixelRGB));
        memcpy(foreground, foreground_initial, N * sizeof(PixelRGBA));
        state.ResumeTiming();

        for (int i = 0; i < N; i++) {
            background[i] = op(foreground[i], background[i]);
        }

        benchmark::DoNotOptimize(background);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(uint64_t(state.iterations()));
}

void BM_composit_scalar_rgba_over_rgba_straight_f32(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgba_generic<F32>(state, blend_RGBAF32_over_RGBAF32_straight_branchless);
}

void BM_composit_scalar_rgba_over_rgba_premultiplied_f32(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgba_generic<F32>(state, blend_RGBAF32_over_RGBAF32_premultiplied_branchless);
}

void BM_composit_scalar_rgba_over_rgba_premultiplied_u8(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgba_generic<U8>(state, blend_RGBAU8_over_RGBAU8_premultiplied_scalar_divide);
}

void BM_composit_scalar_rgba_over_rgb_straight_f32(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgb_generic<F32>(state, blend_RGBAF32_over_RGBF32_straight_branchless);
}

void BM_composit_scalar_rgba_over_rgb_premultiplied_f32(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgb_generic<F32>(state, blend_RGBAF32_over_RGBF32_premultiplied_branchless);
}

void BM_composit_scalar_rgba_over_rgb_premultiplied_u8(benchmark::State& state) {
    BM_composit_scalar_rgba_over_rgb_generic<U8>(state, blend_RGBAU8_over_RGBU8_premultiplied_scalar_divide);
}

BENCHMARK(BM_composit_scalar_rgba_over_rgba_straight_f32)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_composit_scalar_rgba_over_rgba_premultiplied_f32)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_composit_scalar_rgba_over_rgba_premultiplied_u8)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_composit_scalar_rgba_over_rgb_straight_f32)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_composit_scalar_rgba_over_rgb_premultiplied_f32)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_composit_scalar_rgba_over_rgb_premultiplied_u8)->RangeMultiplier(2)->Range(64, 1024);

void temp_f32(benchmark::State& state) {
    RGBA_F32 a = random_generic<RGBA_F32>();
    RGB_F32 b = random_generic<RGB_F32>();

    for (auto _ : state) {
        benchmark::DoNotOptimize(b = blend_RGBAF32_over_RGBF32_premultiplied_branchless(a, b));
    }
}

void temp_u8(benchmark::State& state) {
    RGBA_U8 a = random_generic<RGBA_U8>();
    RGB_U8 b = random_generic<RGB_U8>();

    for (auto _ : state) {
        benchmark::DoNotOptimize(b = blend_RGBAU8_over_RGBU8_premultiplied_sse(a, b));
    }
}

BENCHMARK(temp_f32);
BENCHMARK(temp_u8);
