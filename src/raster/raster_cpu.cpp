#include "raster/rasterization.h"
#include "raster/composit.h"

#include "geometry/barycentric.h"

#include <algorithm>

namespace raster {
    void rasterize_triangles_rgba_2d_cpu_pointwise(const raster::RasterScene &scene, io::Image<io::RGBA255>& image) {
        std::fill(image.data.begin(), image.data.end(), scene.background_colour);

        for (int x = 0; x < image.width; x++) {
            for (int y = 0; y < image.height; y++) {
                float u = (x + 0.5f) / (float)image.width;
                float v = (y + 0.5f) / (float)image.height;
                for (int i = 0; i < scene.triangles.size(); i++) {
                    auto tri = scene.triangles[i];
                    auto bary = geometry2d::barycentric_coordinates({u, v}, tri);
                    if (bary.u >= 0 && bary.v >= 0 && bary.w >= 0) {
                        image.data[x + y * image.width] = composit_over_straight_255(image.data[x + y * image.width], scene.colours[i]);
                    }
                }
            }
        }
    }

    void rasterize_triangles_rgba_2d_cpu_bounded(const raster::RasterScene &scene, io::Image<io::RGBA255>& image) {
        std::fill(image.data.begin(), image.data.end(), scene.background_colour);

        for (int i = 0; i < scene.triangles.size(); i++) {
            auto tri = scene.triangles[i];

            int lower_x = std::max(0, (int)(std::min({tri.a.x, tri.b.x, tri.c.x}) * image.width));
            int lower_y = std::max(0, (int)(std::min({tri.a.y, tri.b.y, tri.c.y}) * image.height));
            int upper_x = std::min(image.width - 1, (int)(std::max({tri.a.x, tri.b.x, tri.c.x}) * image.width));
            int upper_y = std::min(image.height - 1, (int)(std::max({tri.a.y, tri.b.y, tri.c.y}) * image.height));

            for (int x = lower_x; x <= upper_x; x++) {
                for (int y = lower_y; y <= upper_y; y++) {
                    float u = (x + 0.5f) / (float)image.width;
                    float v = (y + 0.5f) / (float)image.height;
                    auto bary = geometry2d::barycentric_coordinates({u, v}, tri);
                    if (bary.u >= 0 && bary.v >= 0 && bary.w >= 0) {
                        image.data[x + y * image.width] = composit_over_straight_255(image.data[x + y * image.width], scene.colours[i]);
                    }
                }
            }
        }
    }

    // https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer/
    void rasterize_triangles_rgba_2d_cpu_integer(const raster::RasterScene &scene, io::Image<io::RGBA255>& image) {
        std::fill(image.data.begin(), image.data.end(), scene.background_colour);

        for (int i = 0; i < scene.triangles.size(); i++) {
            auto tri = scene.triangles[i];

            int xs[3] = {int(tri.a.x * image.width), int(tri.b.x * image.width), int(tri.c.x * image.width)};
            int ys[3] = {int(tri.a.y * image.height), int(tri.b.y * image.height), int(tri.c.y * image.height)};

            // Orient the triangle correctly
            int w = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]);
            if (w < 0) {
                std::swap(xs[0], xs[1]);
                std::swap(ys[0], ys[1]);
            }

            auto orient2d = [](int ax, int ay, int bx, int by, int cx, int cy) {
                return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            };

            int lower_x = std::max(0, std::min({xs[0], xs[1], xs[2]}));
            int lower_y = std::max(0, std::min({ys[0], ys[1], ys[2]}));
            int upper_x = std::min(image.width - 1, std::max({xs[0], xs[1], xs[2]}));
            int upper_y = std::min(image.height - 1, std::max({ys[0], ys[1], ys[2]}));

            int w0_row = orient2d(xs[1], ys[1], xs[2], ys[2], lower_x, lower_y);
            int w1_row = orient2d(xs[2], ys[2], xs[0], ys[0], lower_x, lower_y);
            int w2_row = orient2d(xs[0], ys[0], xs[1], ys[1], lower_x, lower_y);

            int A01 = ys[0] - ys[1], B01 = xs[1] - xs[0];
            int A12 = ys[1] - ys[2], B12 = xs[2] - xs[1];
            int A20 = ys[2] - ys[0], B20 = xs[0] - xs[2];

            // Rasterize
            for (int y = lower_y; y <= upper_y; y++) {
                int w0 = w0_row;
                int w1 = w1_row;
                int w2 = w2_row;

                for (int x = lower_x; x <= upper_x; x++) {
                    // If p is on or inside all edges, render pixel.
                    if ((w0 | w1 | w2) >= 0) {
                        image.data[x + y * image.width] = composit_over_straight_255(image.data[x + y * image.width], scene.colours[i]);
                    }

                    // One step to the right
                    w0 += A12;
                    w1 += A20;
                    w2 += A01;
                }

                // One row step
                w0_row += B12;
                w1_row += B20;
                w2_row += B01;
            }
        }
    }
};
