#pragma once

#include <cassert>

#ifdef __CUDACC__
// CUDA-specific code

#define CUDA_COMPILER
#undef NO_CUDA_COMPILER

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cublas_v2.h>

#define CUDA_CHECK(ans) { cuda_check((ans), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Failure: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#ifndef cublasSafeCall
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line, bool abort = true) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n", __FILE__, __LINE__, err);
        if (abort) exit(err);
    }
}

#else
// Non-CUDA-specific code

#define NO_CUDA_COMPILER
#undef CUDA_COMPILER

#define __host__
#define __device__

#endif

#define PURE [[nodiscard]] __host__ __device__

inline void debug_check_ptr(const void* ptr) {
#ifdef NDEBUG
    (void)ptr;
#else
    assert(ptr);
#endif
}

inline void debug_check_range(int x, int min, int max) {
#ifdef NDEBUG
    (void)x;
    (void)min;
    (void)max;
#else
    assert(x >= min && x < max);
#endif
}
