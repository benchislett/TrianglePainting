#pragma once

#include "common.h"

template<typename T>
struct RGBA {
    T r, g, b, a;
};

template<typename T>
struct RGB {
    T r, g, b;
};

using RGBA_U8 = RGBA<U8>;
using RGB_U8 = RGB<U8>;

using RGBA_I32 = RGBA<I32>;
using RGB_I32 = RGB<I32>;

using RGBA_F32 = RGBA<F32>;
using RGB_F32 = RGB<F32>;
