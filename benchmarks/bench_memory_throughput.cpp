#include <benchmark/benchmark.h>
#include <memory>
#include <cstring>

#include "common.h"
#include "pixel_types.h"
#include "composit.h"

/*  This file benchmarks memory operations on one or two buffers: 
    - memset: fill a buffer with a constant value
    - memcpy: direct copy from one buffer into another
    - vecadd: add two buffers together and store the result in the first buffer
    - - vecadd is parameterized by the data type needing addition, either
        - I32: 32-bit signed integer
        - U8: 8-bit unsigned integer
        - F32: 32-bit float

    The benchmarks are implemented using AVX512 intrinsics for maximum performance.
    The results should give an upper bound on memory throughput on similar operations
    which read zero, one, or two buffers and store into one buffer.

    The sizes of the buffers are parameterized according to small square image shapes:
    - 64x64
    - 128x128
    - 256x256
    - 512x512
    - 1024x1024
    For each resolution (and data type), the correspondingly-sized buffers are allocated
    in aligned memory and filled with random data. 4x the number of pixels are allocated,
    simulating the effect as if used for RGBA images.
*/

/* Use AVX512 memset instead of std::memset */
constexpr bool use_avx512_memset = false;

/* Use AVX512 memcpy instead of std::memcpy */
constexpr bool use_avx512_memcpy = false;

void fast_avx512_memcpy(void * __restrict dest, const void * __restrict src, size_t n_bytes) {
    size_t i = 0;
    for (; i + 64 <= n_bytes; i += 64) {
        __m512i data = _mm512_loadu_si512((U8*)src + i);
        _mm512_storeu_si512((U8*)dest + i, data);
    }
}

void fast_avx512_memset(void * __restrict dest, size_t n_bytes, uint32_t data) {
    size_t i = 0;
    __m512i value = _mm512_set1_epi32(data);
    for (; i + 64 <= n_bytes; i += 64) {
        _mm512_storeu_si512((U8*)dest + i, value);
    }
}

void fast_avx512_vecadd_I32(void * __restrict dest, const void* __restrict src, size_t n_bytes) {
    size_t i = 0;
    for (; i + 64 <= n_bytes; i += 64) {
        __m512i data1 = _mm512_loadu_si512((U8*)src + i);
        __m512i data2 = _mm512_loadu_si512((U8*)dest + i);
        __m512i result = _mm512_add_epi32(data1, data2);
        _mm512_storeu_si512((U8*)dest + i, result);
    }
}

void fast_avx512_vecadd_U8(void * __restrict dest, const void* __restrict src, size_t n_bytes) {
    size_t i = 0;
    for (; i + 64 <= n_bytes; i += 64) {
        __m512i data1 = _mm512_loadu_si512((U8*)src + i);
        __m512i data2 = _mm512_loadu_si512((U8*)dest + i);
        __m512i result = _mm512_add_epi8(data1, data2);
        _mm512_storeu_si512((U8*)dest + i, result);
    }
}

void fast_avx512_vecadd_F32(void * __restrict dest, const void* __restrict src, size_t n_bytes) {
    size_t i = 0;
    for (; i + 64 <= n_bytes; i += 64) {
        __m512 data1 = _mm512_loadu_ps((U8*)src + i);
        __m512 data2 = _mm512_loadu_ps((U8*)dest + i);
        __m512 result = _mm512_add_ps(data1, data2);
        _mm512_storeu_ps((U8*)dest + i, result);
    }
}

template<typename T, typename Op>
void BM_bufferop_image_generic(benchmark::State& state, Op&& op) {
    const int length = state.range(0);
    const int N = length * length * 4; // 4 channels (RGBA)

    T* background = (T*) allocate_aligned(N * sizeof(T));
    T* foreground = (T*) allocate_aligned(N * sizeof(T));

    T* background_initial = (T*) allocate_aligned(N * sizeof(T));
    T* foreground_initial = (T*) allocate_aligned(N * sizeof(T));

    for (int i = 0; i < N; i++) {
        background_initial[i] = random_generic<T>();
        foreground_initial[i] = random_generic<T>();
    }
    
    for (auto _ : state) {
        state.PauseTiming();
        memcpy(background, background_initial, N * sizeof(T));
        memcpy(foreground, foreground_initial, N * sizeof(T));
        state.ResumeTiming();

        op(background, foreground, N);

        benchmark::DoNotOptimize(background);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(uint64_t(state.iterations()));
    state.SetBytesProcessed(uint64_t(N) * uint64_t(state.iterations()) * sizeof(T));
}

template<typename T>
void BM_memset_image_generic(benchmark::State& state) {
    BM_bufferop_image_generic<T>(state, [](T* background, const T* foreground, size_t N) {
        if constexpr (use_avx512_memset) {
            fast_avx512_memset(background, N * sizeof(T), 42);
        } else {
            memset(background, 42, N * sizeof(T));
        }
    });
}

template<typename T>
void BM_memcpy_image_generic(benchmark::State& state) {
    BM_bufferop_image_generic<T>(state, [](T* background, const T* foreground, size_t N) {
        if constexpr (use_avx512_memcpy) {
            fast_avx512_memcpy(background, foreground, N * sizeof(T));
        } else {
            memcpy(background, foreground, N * sizeof(T));
        }
    });
}

template<typename T>
void BM_vecadd_image_generic(benchmark::State& state) {
    BM_bufferop_image_generic<T>(state, [](T* background, const T* foreground, size_t N) {
        if constexpr (std::is_same_v<T, I32>) {
            fast_avx512_vecadd_I32(background, foreground, N * sizeof(T));
        } else if constexpr (std::is_same_v<T, U8>) {
            fast_avx512_vecadd_U8(background, foreground, N * sizeof(T));
        } else if constexpr (std::is_same_v<T, F32>) {
            fast_avx512_vecadd_F32(background, foreground, N * sizeof(T));
        } else {
            fprintf(stderr, "Unsupported type for vecadd operation\n");
            exit(1);
        }
    });
}

BENCHMARK(BM_memset_image_generic<U8>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_memset_image_generic<F32>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_memcpy_image_generic<U8>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_memcpy_image_generic<F32>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_vecadd_image_generic<U8>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_vecadd_image_generic<F32>)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_vecadd_image_generic<I32>)->RangeMultiplier(2)->Range(64, 1024);
