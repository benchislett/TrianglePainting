#include <benchmark/benchmark.h>

#include "impl.h"
#include "utils.h"

#include <random>
#include <immintrin.h>

void BM_memcpy(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 4;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);
    unsigned char* fg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    unsigned char* fg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
        fg_init[i] = random_u8();
    }

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        memcpy(fg, fg_init, N);
        state.ResumeTiming();

        memcpy(bg, fg, N);

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_memcpy)->RangeMultiplier(2)->Range(128, 512);

void vecadd_func(unsigned char* __restrict bg, unsigned char* __restrict fg, int N) {
    for (int i = 0; i < N; i += 64) {
        __m512i bg_el = _mm512_load_si512(bg + i);
        __m512i fg_el = _mm512_load_si512(fg + i);
        __m512i result = _mm512_add_epi8(bg_el, fg_el);
        _mm512_store_si512(bg + i, result);
    }
}

void BM_vecadd(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 4;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);
    unsigned char* fg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    unsigned char* fg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
        fg_init[i] = random_u8();
    }

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        memcpy(fg, fg_init, N);
        state.ResumeTiming();

        vecadd_func(bg, fg, N);

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_vecadd)->RangeMultiplier(2)->Range(128, 512);

void BM_blend_RGBAU8_over_RGBU8_premultiplied_scalar(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 4;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);
    unsigned char* fg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    unsigned char* fg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
        fg_init[i] = random_u8();
    }

    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        memcpy(fg, fg_init, N);
        state.ResumeTiming();

        for (int i = 0; i < N; i += 4) {
            unsigned char fg_r = fg[i + 0];
            unsigned char fg_g = fg[i + 1];
            unsigned char fg_b = fg[i + 2];
            unsigned char fg_a = fg[i + 3];

            unsigned char bg_r = bg[i + 0];
            unsigned char bg_g = bg[i + 1];
            unsigned char bg_b = bg[i + 2];

            unsigned char out_r, out_g, out_b;
            blend_RGBAU8_over_RGBU8_premultiplied_scalar(fg_r, fg_g, fg_b, fg_a,
                                                         bg_r, bg_g, bg_b,
                                                         out_r, out_g, out_b);

            bg[i + 0] = out_r;
            bg[i + 1] = out_g;
            bg[i + 2] = out_b;
        }

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_blend_RGBAU8_over_RGBU8_premultiplied_scalar)->RangeMultiplier(2)->Range(128, 512);

void BM_blend_RGBAU8_over_RGBU8_premultiplied_avx(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 4;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);
    unsigned char* fg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    unsigned char* fg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
        fg_init[i] = random_u8();
    }

    
    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        memcpy(fg, fg_init, N);
        state.ResumeTiming();

        
        for (size_t i = 0; i < N; i += 128) {
            __m256i fg_r8 = _mm256_load_si256((__m256i const*)(fg + i +  0));
            __m256i fg_g8 = _mm256_load_si256((__m256i const*)(fg + i + 32));
            __m256i fg_b8 = _mm256_load_si256((__m256i const*)(fg + i + 64));
            __m256i fg_a8 = _mm256_load_si256((__m256i const*)(fg + i + 96));

            __m256i bg_r8 = _mm256_load_si256((__m256i const*)(bg + i +   0));
            __m256i bg_g8 = _mm256_load_si256((__m256i const*)(bg + i +  32));
            __m256i bg_b8 = _mm256_load_si256((__m256i const*)(bg + i +  64));

            const __m256i zero = _mm256_setzero_si256();
            auto widen  = [&](const __m256i& v8){
                return std::pair<__m256i,__m256i>{
                    _mm256_unpacklo_epi8(v8, zero),  // low 16
                    _mm256_unpackhi_epi8(v8, zero)   // high 16
                };
            };
            auto [fg_r16L, fg_r16H] = widen(fg_r8);
            auto [fg_g16L, fg_g16H] = widen(fg_g8);
            auto [fg_b16L, fg_b16H] = widen(fg_b8);
            auto [fg_a16L, fg_a16H] = widen(fg_a8);

            auto [bg_r16L, bg_r16H] = widen(bg_r8);
            auto [bg_g16L, bg_g16H] = widen(bg_g8);
            auto [bg_b16L, bg_b16H] = widen(bg_b8);

            __m256i out_r16L, out_g16L, out_b16L, out_r16H, out_g16H, out_b16H;
            blend_RGBAU16x16_over_RGBU16x16_premultiplied_avx(
                fg_r16L, fg_g16L, fg_b16L, fg_a16L,
                bg_r16L, bg_g16L, bg_b16L,
                out_r16L, out_g16L, out_b16L
            );
            blend_RGBAU16x16_over_RGBU16x16_premultiplied_avx(
                fg_r16H, fg_g16H, fg_b16H, fg_a16H,
                bg_r16H, bg_g16H, bg_b16H,
                out_r16H, out_g16H, out_b16H
            );

            __m256i out_r8L, out_g8L, out_b8L, out_r8H, out_g8H, out_b8H;
            out_r8L = _mm256_packus_epi16(out_r16L, out_r16L);
            out_g8L = _mm256_packus_epi16(out_g16L, out_g16L);
            out_b8L = _mm256_packus_epi16(out_b16L, out_b16L);
            out_r8H = _mm256_packus_epi16(out_r16H, out_r16H);
            out_g8H = _mm256_packus_epi16(out_g16H, out_g16H);
            out_b8H = _mm256_packus_epi16(out_b16H, out_b16H);

            // merge low+high 128-bit halves so we can store once per channel
            __m256i out_r8 = _mm256_permute2x128_si256(out_r8L, out_r8H, 0x20);
            __m256i out_g8 = _mm256_permute2x128_si256(out_g8L, out_g8H, 0x20);
            __m256i out_b8 = _mm256_permute2x128_si256(out_b8L, out_b8H, 0x20);

            // ---- 4. 32-byte stores -------------------------------------------------
            _mm256_store_si256((__m256i*)(bg + i +  0), out_r8);
            _mm256_store_si256((__m256i*)(bg + i + 32), out_g8);
            _mm256_store_si256((__m256i*)(bg + i + 64), out_b8);
        }
        

        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_blend_RGBAU8_over_RGBU8_premultiplied_avx)->RangeMultiplier(2)->Range(128, 512);

void BM_blend_RGBAU8_over_RGBU8_premultiplied_avx512(benchmark::State& state) {
    const int N = state.range(0) * state.range(0) * 4;
    unsigned char* bg = (unsigned char*)allocate_aligned(N);
    unsigned char* fg = (unsigned char*)allocate_aligned(N);

    unsigned char* bg_init = (unsigned char*)allocate_aligned(N);
    unsigned char* fg_init = (unsigned char*)allocate_aligned(N);
    for (int i = 0; i < N; ++i) {
        bg_init[i] = random_u8();
        fg_init[i] = random_u8();
    }

    
    for (auto _ : state) {
        state.PauseTiming();
        memcpy(bg, bg_init, N);
        memcpy(fg, fg_init, N);
        state.ResumeTiming();

        size_t i_fg = 0;
        size_t i_bg = 0;
        for (size_t i_fg = 0; i_fg < N; i_fg += 128)          // 32 pixels / tile
        {
            /* ---- 1. 32-byte loads (AVX-256) ------------------------------------ */
            __m256i fg_r8 = _mm256_load_si256((__m256i const*)(fg + i_fg +  0));
            __m256i fg_g8 = _mm256_load_si256((__m256i const*)(fg + i_fg + 32));
            __m256i fg_b8 = _mm256_load_si256((__m256i const*)(fg + i_fg + 64));
            __m256i fg_a8 = _mm256_load_si256((__m256i const*)(fg + i_fg + 96));

            __m256i bg_r8 = _mm256_load_si256((__m256i const*)(bg + i_bg +   0));
            __m256i bg_g8 = _mm256_load_si256((__m256i const*)(bg + i_bg +  32));
            __m256i bg_b8 = _mm256_load_si256((__m256i const*)(bg + i_bg +  64));

            /* ---- 2. widen 32×u8  → 32×u16   (VPMOVZXBW) ------------------------ */

            __m512i fg_r16 = _mm512_cvtepu8_epi16(fg_r8);
            __m512i fg_g16 = _mm512_cvtepu8_epi16(fg_g8);
            __m512i fg_b16 = _mm512_cvtepu8_epi16(fg_b8);
            __m512i fg_a16 = _mm512_cvtepu8_epi16(fg_a8);

            __m512i bg_r16 = _mm512_cvtepu8_epi16(bg_r8);
            __m512i bg_g16 = _mm512_cvtepu8_epi16(bg_g8);
            __m512i bg_b16 = _mm512_cvtepu8_epi16(bg_b8);

            /* ---- 3.  core blend  ---------------------------------------------- */

            __m512i out_r16, out_g16, out_b16;
            blend_RGBAU16x32_over_RGBU16x32_premultiplied_avx512(
                fg_r16, fg_g16, fg_b16, fg_a16,
                bg_r16, bg_g16, bg_b16,
                out_r16, out_g16, out_b16
            );

            /* ---- 4.  pack-to-u8 *and* store   (VPMOVUSWB) --------------------- */
            _mm512_mask_cvtusepi16_storeu_epi8(bg + i_bg +  0, 0xFFFFFFFFu, out_r16); // R
            _mm512_mask_cvtusepi16_storeu_epi8(bg + i_bg + 32, 0xFFFFFFFFu, out_g16); // G
            _mm512_mask_cvtusepi16_storeu_epi8(bg + i_bg + 64, 0xFFFFFFFFu, out_b16); // B

            i_bg += 96;
        }
        
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * N);
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_blend_RGBAU8_over_RGBU8_premultiplied_avx512)->RangeMultiplier(2)->Range(128, 512);