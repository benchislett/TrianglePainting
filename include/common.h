#pragma once

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

#define PURE [[nodiscard]] __host__ __device__
