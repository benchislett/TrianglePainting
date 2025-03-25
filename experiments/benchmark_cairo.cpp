#include <iostream>
#include <chrono>
#include <vector>
#include <cairo/cairo.h>

#include "../include/image.h"
#include "benchmark_triangle_rasterization.h"

struct CairoRasterImpl : public RasterImpl {
    cairo_t *cr;
    cairo_surface_t *surface;
    ImageView<RGBA255> im;

    void set_canvas(ImageView<RGBA255> background) override {
        surface = cairo_image_surface_create_for_data((unsigned char*) background.data(), CAIRO_FORMAT_ARGB32, background.width(), background.height(), background.width() * 4);
        cr = cairo_create(surface);
        im = background;
        cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    }

    void render(SampleInput sample) override {
        int coords[6] = {
            int(sample.triangle[0] * im.width()),
            int(sample.triangle[1] * im.height()),
            int(sample.triangle[2] * im.width()),
            int(sample.triangle[3] * im.height()),
            int(sample.triangle[4] * im.width()),
            int(sample.triangle[5] * im.height())
        };
        cairo_set_source_rgba(cr, 
            sample.colour_rgba[0] / 255.0, 
            sample.colour_rgba[1] / 255.0, 
            sample.colour_rgba[2] / 255.0, 
            sample.colour_rgba[3] / 255.0);
        cairo_move_to(cr, coords[0], coords[1]);
        cairo_line_to(cr, coords[2], coords[3]);
        cairo_line_to(cr, coords[4], coords[5]);
        cairo_close_path(cr);
        cairo_fill(cr);
    }
};

int main () {
    auto impl = std::make_shared<CairoRasterImpl>();
    default_benchmark_main(impl);
    return 0;
}
