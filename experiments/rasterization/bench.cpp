#include <benchmark/benchmark.h>

#include "impl.h"
#include "utils.h"

#include <random>
#include <immintrin.h>

void BM_draw_fill_rgba_over_imrgb_scalar(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 3;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
    }

    unsigned char fg[4] = {random_u8(), random_u8(), random_u8(), random_u8()};

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        state.ResumeTiming();

        draw_fill_rgba_over_imrgb_scalar(bg, fg, state.range(0));
        
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_draw_fill_rgba_over_imrgb_scalar)->RangeMultiplier(2)->Range(128, 512);

void BM_draw_fill_rgba_over_imrgb_avx512(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 3;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
    }

    unsigned char fg[4] = {random_u8(), random_u8(), random_u8(), random_u8()};

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        state.ResumeTiming();

        draw_fill_rgba_over_imrgb_avx512(bg, fg, state.range(0));
        
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_draw_fill_rgba_over_imrgb_avx512)->RangeMultiplier(2)->Range(128, 512);

void BM_draw_triangle_rgba_over_imrgb_scalar(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 3;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
    }

    unsigned char fg[4] = {random_u8(), random_u8(), random_u8(), random_u8()};

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        state.ResumeTiming();

        draw_triangle_rgba_over_imrgb_scalar(bg, fg, state.range(0));
        
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_draw_triangle_rgba_over_imrgb_scalar)->RangeMultiplier(2)->Range(128, 512);

void BM_draw_triangle_rgba_over_imrgb_avx512(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 3;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
    }

    unsigned char fg[4] = {random_u8(), random_u8(), random_u8(), random_u8()};

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        state.ResumeTiming();

        draw_triangle_rgba_over_imrgb_avx512(bg, fg, state.range(0));
        
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_draw_triangle_rgba_over_imrgb_avx512)->RangeMultiplier(2)->Range(128, 512);
