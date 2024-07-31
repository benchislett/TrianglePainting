#pragma once

#include "geometry/types.h"
#include "io/image.h"

#include <vector>

namespace raster {
    void rasterize_triangles_rgba_2d_cpu_pointwise(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<io::RGBA255>& colours,
        io::Image<io::RGBA255>& image);

    void rasterize_triangles_rgba_2d_cpu_bounded(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<io::RGBA255>& colours,
        io::Image<io::RGBA255>& image);

    void rasterize_triangles_rgba_2d_opengl(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<io::RGBA255>& colours,
        io::Image<io::RGBA255>& image);
};
