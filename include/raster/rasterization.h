#pragma once

#include "geometry/types.h"
#include "io/image.h"

#include <vector>

namespace raster {

    struct RGB01 {
        float r, g, b;
    };

    struct RGBA01 {
        float r, g, b, a;
    };

    struct RGB255 {
        unsigned char r, g, b;
    };

    struct RGBA255 {
        unsigned char r, g, b, a;
    };

    template<typename ColourT>
    struct ImageBuffer {
        std::vector<ColourT> data;
        int width, height;
    };

    inline io::Image cast_image(const ImageBuffer<RGBA255>& image) {
        io::Image out;
        out.data.resize(image.data.size() * 4);
        for (int i = 0; i < image.data.size(); i++) {
            out.data[i * 4 + 0] = image.data[i].r;
            out.data[i * 4 + 1] = image.data[i].g;
            out.data[i * 4 + 2] = image.data[i].b;
            out.data[i * 4 + 3] = image.data[i].a;
        }
        out.width = image.width;
        out.height = image.height;
        out.channels = 4;
        return out;
    }

    inline io::Image cast_image(const ImageBuffer<RGB255>& image) {
        io::Image out;
        out.data.resize(image.data.size() * 3);
        for (int i = 0; i < image.data.size(); i++) {
            out.data[i * 3 + 0] = image.data[i].r;
            out.data[i * 3 + 1] = image.data[i].g;
            out.data[i * 3 + 2] = image.data[i].b;
        }
        out.width = image.width;
        out.height = image.height;
        out.channels = 3;
        return out;
    }

    void rasterize_triangles_rgba_2d_cpu_pointwise(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<RGBA255>& colours,
        ImageBuffer<RGBA255>& image);

    void rasterize_triangles_rgba_2d_cpu_bounded(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<RGBA255>& colours,
        ImageBuffer<RGBA255>& image);

    void rasterize_triangles_rgba_2d_opengl(
        const std::vector<geometry2d::triangle>& triangles,
        const std::vector<RGBA255>& colours,
        ImageBuffer<RGBA255>& image);
};
