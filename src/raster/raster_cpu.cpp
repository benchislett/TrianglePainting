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
};
