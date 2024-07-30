#include "geometry/barycentric.h"
#include "io/image.h"

#include "lodepng.h"

#include <vector>

// Draw a triangle, colour it using barycentric coordinates, and save the result to a PNG file.
int main(int argc, char** argv) {
    unsigned int width = 512;
    if (argc > 1) {
        width = std::atoi(argv[1]);
    }
    std::string output_filename = "raster_triangle_barycentric.png";
    if (argc > 2) {
		output_filename = argv[2];
	}

    std::vector<unsigned char> image(width * width * 4, 0);

    geometry2d::triangle tri1{{0.25, 0.25}, {0.75, 0.25}, {0.5, 0.75}};
    geometry2d::triangle tri2{{0.1, 0.2}, {0.15, 0.3}, {0.3, 0.15}};

    std::vector<geometry2d::triangle> tris{tri1, tri2};

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            float u = (x + 0.5f) / (float)width;
            float v = (y + 0.5f) / (float)width;
            for (auto tri : tris) {
                auto bary = geometry2d::barycentric_coordinates({u, v}, tri);
                if (bary.u >= 0 && bary.v >= 0 && bary.w >= 0) {
                    image[(x + y * width) * 4 + 0] = (unsigned int)(bary.u * 255);
                    image[(x + y * width) * 4 + 1] = (unsigned int)(bary.v * 255);
                    image[(x + y * width) * 4 + 2] = (unsigned int)(bary.w * 255);
                    image[(x + y * width) * 4 + 3] = 255;
                    break;
                }
            }
        }
    }

    io::save_png(output_filename, {image, width, width, 4});

    return 0;
}