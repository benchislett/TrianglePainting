#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "io/image.h"
#include "io/png.h"
#include "geometry/types.h"
#include "raster/rasterization.h"

#include <nlohmann/json.hpp>

struct SceneParams { std::vector<geometry2d::triangle> triangles; std::vector<io::RGBA255> colours; io::RGBA255 background; };

auto load_json(const std::string& filename) {
    auto json = nlohmann::json::parse(std::ifstream(filename));
    std::vector<geometry2d::triangle> triangles;
    std::vector<io::RGBA255> colours;
    io::RGBA01 background{float(json["background"][0]), float(json["background"][1]), float(json["background"][2]), float(json["background"][3])};
    for (auto& tri : json["triangles"]) {
        for (int i = 0; i < 3; i++) {
            triangles.push_back(geometry2d::triangle{
                {tri["vertices"][0][0], tri["vertices"][0][1]},
                {tri["vertices"][1][0], tri["vertices"][1][1]},
                {tri["vertices"][2][0], tri["vertices"][2][1]}
            });
            io::RGBA01 colour{float(tri["colour"][0]), float(tri["colour"][1]), float(tri["colour"][2]), float(tri["colour"][3])};
            colours.push_back(io::to_rgba255(colour));
        }
    }
    return SceneParams{triangles, colours, io::to_rgba255(background)};
}

int main(int argc, char** argv)
{
	/* Config options */
	int image_resolution = 512;
    std::string input_filename = "random_triangles.json";
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

    SceneParams scene;
    try {
        auto [triangles, colours, background] = load_json(input_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading JSON, check that the format is correct. Error message: " << e.what() << std::endl;
        return 1;
    }

    io::Image<io::RGBA255> image;
    image.width = image_resolution;
    image.height = image_resolution;
    image.data.resize(image.width * image.height);
    std::fill(image.data.begin(), image.data.end(), io::RGBA255{0, 0, 0, 0});

    raster::rasterize_triangles_rgba_2d_cpu_pointwise(scene.triangles, scene.colours, image);

    io::save_png_rgba(output_filename, image);

	return 0;
}
