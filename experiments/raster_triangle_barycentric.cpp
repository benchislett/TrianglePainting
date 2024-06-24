#include "geometry/barycentric.h"
#include "io/image.h"

#include "lodepng.h"

#include <vector>

// Draw a triangle, colour it using barycentric coordinates, and save the result to a PNG file.
int main() {
    const int width = 512;
    std::vector<unsigned char> image(width * width * 4, 255);

    geometry2d::triangle tri{{0.25, 0.25}, {0.75, 0.25}, {0.5, 0.75}};

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            float u = (x + 0.5f) / (float)width;
            float v = (y + 0.5f) / (float)width;
            auto bary = geometry2d::barycentric_coordinates({u, v}, tri);
            if (bary.u >= 0 && bary.v >= 0 && bary.w >= 0) {
                image[(x + y * width) * 4 + 0] = (unsigned int)(bary.u * 255);
                image[(x + y * width) * 4 + 1] = (unsigned int)(bary.v * 255);
                image[(x + y * width) * 4 + 2] = (unsigned int)(bary.w * 255);
                image[(x + y * width) * 4 + 3] = 255;
            }
        }
    }

    io::save_png("raster_triangle_barycentric.png", {image, width, width, 4});

    return 0;
}