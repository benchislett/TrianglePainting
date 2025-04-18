#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <immintrin.h>
#include <xmmintrin.h>

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

using F32 = float;
using F64 = double;

void* allocate_aligned(size_t size, size_t alignment = 4096) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        fprintf(stderr, "Aligned memory allocation failed\n");
        return nullptr;
    }
    return ptr;
}
