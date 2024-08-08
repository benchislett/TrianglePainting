#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <chrono>

#include "io/image.h"
#include "io/png.h"
#include "geometry/types.h"
#include "raster/rasterization.h"

#include "triangle_image_state.h"

int main(int argc, char** argv)
{
	/* Config options */
	int image_resolution = 512;
    std::string input_filename = "random_triangles.json";
    std::string mode = "pointwise";
	if (argc > 1) {
		std::string input_filename = argv[1];
	}
	if (argc > 2) {
		image_resolution = std::atoi(argv[2]);
	}
	std::string output_filename = "output.png";
	if (argc > 3) {
		output_filename = argv[3];
	}
    if (argc > 4) {
        mode = argv[4];
    }

    RasterScene scene;
    try {
        scene = load_json(input_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading JSON, check that the format is correct. Error message: " << e.what() << std::endl;
        return 1;
    }

    io::Image<io::RGBA255> image;
    image.width = image_resolution;
    image.height = image_resolution;
    image.data.resize(image.width * image.height);
    std::fill(image.data.begin(), image.data.end(), io::RGBA255{0, 0, 0, 255});

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "opengl") {
        raster::rasterize_triangles_to_image_opengl(scene.triangles, scene.colours, scene.background_colour, image);
    } else {
        for (int i = 0; i < scene.triangles.size(); i++) {
            auto& tri = scene.triangles[i];
            auto& colour = scene.colours[i];
            if (mode == "bounded") {
                raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::Bounded>(tri, colour, image);
            } else if (mode == "integer") {
                raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::Integer>(tri, colour, image);
            } else if (mode == "polygon" || mode == "scanline") {
                raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::ScanlinePolygon>(tri, colour, image);
            } else {
                fprintf(stderr, "Invalid rasterization mode selected\n");
                exit (1);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    io::save_png_rgba(output_filename, image);

	return 0;
}
