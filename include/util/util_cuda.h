#pragma once

#ifdef __CUDACC__
#define MISC_SYMS __host__ __device__ inline

#include <cuda/std/array>
#include <cuda/std/tuple>

template<typename T, int N>
using array = cuda::std::array<T, N>;

template<typename T1, typename T2>
using pair = cuda::std::pair<T1, T2>;

using cuda::std::make_pair;

#else
#define MISC_SYMS inline
#include <cmath>
#include <array>
#include <tuple>
#include <utility>

struct float2 {
    float x, y;
};
struct float3 {
    float x, y, z;
};
struct float4 {
    float x, y, z, w;
};

MISC_SYMS float2 make_float2(float x, float y) { return float2{x, y}; }
MISC_SYMS float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
MISC_SYMS float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

template<typename T, int N>
using array = std::array<T, N>;

template<typename T1, typename T2>
using pair = std::pair<T1, T2>;

using std::make_pair;

#endif