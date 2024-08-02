#pragma once

#include "io/image.h"

namespace raster {
    io::RGBA01 composit_over_premultiplied_01(const io::RGBA01& background, const io::RGBA01 &foreground);
    io::RGBA01 composit_over_straight_01(const io::RGBA01& background, const io::RGBA01 &foreground);

    io::RGBA255 composit_over_premultiplied_255(const io::RGBA255& background, const io::RGBA255 &foreground);
    io::RGBA255 composit_over_straight_255(const io::RGBA255& background, const io::RGBA255 &foreground);
};
