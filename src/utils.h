#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>

inline void* allocate_aligned(size_t size, size_t alignment = 4096) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        std::cerr << "Aligned memory allocation failed" << std::endl;
        return nullptr;
    }
    return ptr;
}
