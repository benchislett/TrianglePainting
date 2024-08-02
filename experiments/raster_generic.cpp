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

#include <nlohmann/json.hpp>

auto load_json(const std::string& filename, bool premultiply) {
    auto json = nlohmann::json::parse(std::ifstream(filename));
    std::vector<geometry2d::triangle> triangles;
    std::vector<io::RGBA255> colours;
    io::RGBA01 background{float(json["background"][0]), float(json["background"][1]), float(json["background"][2]), float(json["background"][3])};
    for (auto& tri : json["triangles"]) {
        triangles.push_back(geometry2d::triangle{
            {tri["vertices"][0][0], tri["vertices"][0][1]},
            {tri["vertices"][1][0], tri["vertices"][1][1]},
            {tri["vertices"][2][0], tri["vertices"][2][1]}
        });
        io::RGBA01 colour{float(tri["colour"][0]), float(tri["colour"][1]), float(tri["colour"][2]), float(tri["colour"][3])};
        if (premultiply) {
            colour.r = colour.r * colour.a;
            colour.g = colour.g * colour.a;
            colour.b = colour.b * colour.a;
        }
        colours.push_back(io::to_rgba255(colour));
    }
    return raster::RasterScene{triangles, colours, io::to_rgba255(background)};
}

int main(int argc, char** argv)
{
	/* Config options */
	int image_resolution = 512;
    std::string input_filename = "random_triangles.json";
    bool premultiply = false;
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

    raster::RasterScene scene;
    try {
        scene = load_json(input_filename, premultiply);
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

    if (mode == "bounded") {
        raster::rasterize_triangles_rgba_2d_cpu_bounded(scene, image);
    } else if (mode == "opengl") {
        raster::rasterize_triangles_rgba_2d_opengl(scene, image);
    } else {
        raster::rasterize_triangles_rgba_2d_cpu_pointwise(scene, image);
    }

    if (premultiply) {
        for (auto& pixel : image.data) {
            if (pixel.a > 0) {
                io::RGBA01 pixel01 = io::to_rgba01(pixel);
                pixel01.r = pixel01.r / pixel01.a;
                pixel01.g = pixel01.g / pixel01.a;
                pixel01.b = pixel01.b / pixel01.a;
                pixel = io::to_rgba255(pixel01);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    io::save_png_rgba(output_filename, image);

	return 0;
}
