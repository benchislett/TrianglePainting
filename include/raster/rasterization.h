#pragma once

#include "geometry/types.h"
#include "io/image.h"

#include <vector>

namespace raster {
    struct RasterScene {
        std::vector<geometry2d::triangle> triangles;
        std::vector<io::RGBA255> colours;
        io::RGBA255 background_colour;
    };

    void rasterize_triangles_rgba_2d_cpu_pointwise(const RasterScene& scene, io::Image<io::RGBA255>& image);
    void rasterize_triangles_rgba_2d_cpu_bounded(const RasterScene& scene, io::Image<io::RGBA255>& image);
    void rasterize_triangles_rgba_2d_cpu_integer(const RasterScene& scene, io::Image<io::RGBA255>& image);
    void rasterize_triangles_rgba_2d_opengl(const RasterScene& scene, io::Image<io::RGBA255>& image);
};
