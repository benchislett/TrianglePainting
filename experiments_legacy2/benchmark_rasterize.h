#pragma once

#include <iostream>
#include <chrono>
#include <vector>

#include "rasterize.h"
#include "shaders.h"
#include "image.h"
#include "colours.h"

#include "benchmark_triangle_rasterization.h"

template<RasterStrategy strategy>
struct PolypaintRasterImpl : public RasterImpl {
    ImageView<RGBA255> im;

    void set_canvas(ImageView<RGBA255> background) override {
        im = background;
    }

    void render(SampleInput sample) override {
        Triangle triangle;
        triangle.vertices = {
            Point{sample.triangle[0], sample.triangle[1]},
            Point{sample.triangle[2], sample.triangle[3]},
            Point{sample.triangle[4], sample.triangle[5]}
        };
        CompositOverShader shader(im, RGBA255{sample.colour_rgba[0], sample.colour_rgba[1], sample.colour_rgba[2], sample.colour_rgba[3]});
        RasterConfig config;
        config.strategy = strategy;
        config.image_width = im.width();
        config.image_height = im.height();
        rasterize(std::make_shared<Triangle>(triangle), shader, config);
    }

    std::string name() const override {
        return "RasterDispatch " + raster_strategy_name(strategy);
    }
};
