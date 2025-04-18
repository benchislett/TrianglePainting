#pragma once

#include "common.h"

#include <random>

/* Generic Pixel Types */

template<typename T>
struct RGBA {
    T r, g, b, a;
};

template<typename T>
struct RGB {
    T r, g, b;
};

/* Specialized Pixel Types */

using RGBA_U8 = RGBA<U8>;
using RGB_U8 = RGB<U8>;

using RGBA_I32 = RGBA<I32>;
using RGB_I32 = RGB<I32>;

using RGBA_F32 = RGBA<F32>;
using RGB_F32 = RGB<F32>;

/* Random initializers for specialized pixel types */

F32 random_F32(F32 min, F32 max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<F32> dis(min, max);
    return dis(gen);
}

I32 random_I32(I32 min, I32 max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<I32> dis(min, max);
    return dis(gen);
}

template<typename T>
T random_generic() {
    // return a random generic type,
    // or populate a pixel type with random values of the subtype
    if constexpr (std::is_same_v<T, I32>) {
        return random_I32(1, 255);
    } else if constexpr (std::is_same_v<T, F32>) {
        return random_F32(0.01f, 0.99f);
    } else if constexpr (std::is_same_v<T, U8>) {
        return (U8)random_I32(1, 255);
    } else if constexpr (std::is_same_v<T, RGBA_F32>) {
        return RGBA_F32{
            random_generic<F32>(),
            random_generic<F32>(),
            random_generic<F32>(),
            random_generic<F32>()
        };
    } else if constexpr (std::is_same_v<T, RGB_F32>) {
        return RGB_F32{
            random_generic<F32>(),
            random_generic<F32>(),
            random_generic<F32>()
        };
    } else if constexpr (std::is_same_v<T, RGBA_U8>) {
        return RGBA_U8{
            random_generic<U8>(),
            random_generic<U8>(),
            random_generic<U8>(),
            random_generic<U8>()
        };
    } else if constexpr (std::is_same_v<T, RGB_U8>) {
        return RGB_U8{
            random_generic<U8>(),
            random_generic<U8>(),
            random_generic<U8>()
        };
    } else if constexpr (std::is_same_v<T, RGBA_I32>) {
        return RGBA_I32{
            random_generic<I32>(),
            random_generic<I32>(),
            random_generic<I32>(),
            random_generic<I32>()
        };
    } else if constexpr (std::is_same_v<T, RGB_I32>) {
        return RGB_I32{
            random_generic<I32>(),
            random_generic<I32>(),
            random_generic<I32>()
        };
    } else {
        static_assert("Unsupported type");
        return T{};
    }
}
