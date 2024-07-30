#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "io/image.h"
#include "geometry/types.h"
#include "raster/rasterization.h"

#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
	/* Config options */
	int image_resolution = 512;
    std::vector<geometry2d::triangle> triangles;
    std::vector<raster::RGBA255> colours;
    raster::ImageBuffer<raster::RGBA255> image;
    image.width = image_resolution;
    image.height = image_resolution;
    image.data.resize(image.width * image.height);
    std::fill(image.data.begin(), image.data.end(), raster::RGBA255{0, 0, 0, 0});
	if (argc > 1) {
		std::string input_filename = argv[1];
		auto json = nlohmann::json::parse(std::ifstream(input_filename));
		for (auto& tri : json["triangles"]) {
			for (int i = 0; i < 3; i++) {
                triangles.push_back(geometry2d::triangle{
                    {tri["vertices"][0][0], tri["vertices"][0][1]},
                    {tri["vertices"][1][0], tri["vertices"][1][1]},
                    {tri["vertices"][2][0], tri["vertices"][2][1]}
                });
                colours.push_back(raster::RGBA255{
                    (unsigned char)(float(tri["colour"][0]) * 255),
                    (unsigned char)(float(tri["colour"][1]) * 255),
                    (unsigned char)(float(tri["colour"][2]) * 255),
                    (unsigned char)(float(tri["colour"][3]) * 255)
                });
			}
		}
	}
	if (argc > 2) {
		image_resolution = std::atoi(argv[2]);
	}
	std::string output_filename = "output.png";
	if (argc > 3) {
		output_filename = argv[3];
	}

    rasterize_triangles_rgba_2d_opengl(triangles, colours, image);

    io::save_png(output_filename, raster::cast_image(image));

	return 0;
}
