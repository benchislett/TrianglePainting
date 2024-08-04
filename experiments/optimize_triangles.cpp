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

auto load_json(const std::string& filename) {
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
        colours.push_back(io::to_rgba255(colour));
    }
    return raster::RasterScene{triangles, colours, io::to_rgba255(background)};
}

void save_json(const std::string& filename, const raster::RasterScene& scene) {
    nlohmann::json json;
    io::RGBA01 background = io::to_rgba01(scene.background_colour);
    json["background"] = {background.r, background.g, background.b, background.a};

    for (size_t i = 0; i < scene.triangles.size(); i++) {
        auto tri = scene.triangles[i];
        auto colour = io::to_rgba01(scene.colours[i]);
        json["triangles"].push_back({
            {"vertices", {
                {tri.a.x, tri.a.y},
                {tri.b.x, tri.b.y},
                {tri.c.x, tri.c.y}
            }},
            {"colour", {colour.r, colour.g, colour.b, colour.a}}
        });
    }
    std::ofstream(filename) << json.dump(4) << std::endl;
}

io::Image<io::RGBA255> render(const raster::RasterScene& scene, int image_resolution) {
    io::Image<io::RGBA255> image;
    image.width = image_resolution;
    image.height = image_resolution;
    image.data.resize(image.width * image.height);

    raster::rasterize_triangles_rgba_2d_cpu_integer(scene, image);

    return image;
}

int loss(const raster::RasterScene& scene, const io::Image<io::RGBA255>& target) {
    assert (target.width == target.height);
    auto image = render(scene, target.width);

    int acc = 0;
    for (int i = 0; i < target.width * target.height; i++) {
        int delta_r = target.data[i].r - image.data[i].r;
        int delta_g = target.data[i].g - image.data[i].g;
        int delta_b = target.data[i].b - image.data[i].b;
        acc += std::abs(delta_r) + std::abs(delta_g) + std::abs(delta_b);
    }

    return acc;
}

unsigned char next_255(unsigned char x) {
    if (x == 255) return 255;
    else return x + 1;
}

unsigned char prev_255(unsigned char x) {
    if (x == 0) return 0;
    else return x - 1;
}

int main(int argc, char** argv)
{
	/* Config options */
    std::string input_filename = "random_triangles.json";
	if (argc > 1) {
		std::string input_filename = argv[1];
	}
	std::string output_filename = "output.png";
	if (argc > 2) {
		output_filename = argv[2];
	}
    std::string output_config_filename = "new_triangles.json";
    if (argc > 3) {
        output_config_filename = argv[3];
    }
    std::string target_image_filename = "target.png";
    if (argc > 4) {
        target_image_filename = argv[4];
    }

    raster::RasterScene scene;
    try {
        scene = load_json(input_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading JSON, check that the format is correct. Error message: " << e.what() << std::endl;
        return 1;
    }

    auto target = io::load_png_rgba(target_image_filename);

    for (int i = 0; i < 100000; i++) {
        // choose a random triangle
        int idx = rand() % scene.triangles.size();
        auto& tri = scene.triangles[idx];
        auto tri_prev = tri;
        auto& col = scene.colours[idx];
        auto col_prev = col;

        // choose a random vertex
        int vidx = rand() % 3;
        auto& vertex = vidx == 0 ? tri.a : vidx == 1 ? tri.b : tri.c;
        auto vertex_prev = vertex;

        float delta_unit = 1.0 / float(target.width);

        int loss_baseline = loss(scene, target);
        
        vertex.x = vertex_prev.x - delta_unit;
        int loss1 = loss(scene, target);

        vertex.x = vertex_prev.x + delta_unit;
        int loss2 = loss(scene, target);
        vertex.x = vertex_prev.x;

        vertex.y = vertex_prev.y - delta_unit;
        int loss3 = loss(scene, target);

        vertex.y = vertex_prev.y + delta_unit;
        int loss4 = loss(scene, target);
        vertex.y = vertex_prev.y;

        col.r = next_255(col_prev.r);
        int loss5 = loss(scene, target);

        col.r = prev_255(col_prev.r);
        int loss6 = loss(scene, target);
        col.r = col_prev.r;

        col.g = next_255(col_prev.g);
        int loss7 = loss(scene, target);

        col.g = prev_255(col_prev.g);
        int loss8 = loss(scene, target);
        col.g = col_prev.g;

        col.b = next_255(col_prev.b);
        int loss9 = loss(scene, target);

        col.b = prev_255(col_prev.b);
        int loss10 = loss(scene, target);
        col.b = col_prev.b;

        int min_loss_x = std::min({loss1, loss2, loss_baseline});
        int min_loss_y = std::min({loss3, loss4, loss_baseline});
        int min_loss_r = std::min({loss5, loss6, loss_baseline});
        int min_loss_g = std::min({loss7, loss8, loss_baseline});
        int min_loss_b = std::min({loss9, loss10, loss_baseline});
        int min_loss = std::min({min_loss_x, min_loss_y, min_loss_r, min_loss_g, min_loss_b});

        if (min_loss_x == loss1) {
            vertex.x -= delta_unit;
        } else if (min_loss_x == loss2) {
            vertex.x += delta_unit;
        }

        if (min_loss_y == loss3) {
            vertex.y -= delta_unit;
        } else if (min_loss_y == loss4) {
            vertex.y += delta_unit;
        }

        if (min_loss_r == loss5) {
            col.r = next_255(col_prev.r);
        } else if (min_loss_r == loss6) {
            col.r = prev_255(col_prev.r);
        }

        if (min_loss_g == loss7) {
            col.g = next_255(col_prev.g);
        } else if (min_loss_g == loss8) {
            col.g = prev_255(col_prev.g);
        }

        if (min_loss_b == loss9) {
            col.b = next_255(col_prev.b);
        } else if (min_loss_b == loss10) {
            col.b = prev_255(col_prev.b);
        }

        if (i % 100 == 0) {
            printf("%d | %d\n", i, min_loss);
        }
    }

    io::save_png_rgba(output_filename, render(scene, target.width));

    save_json(output_config_filename, scene);

	return 0;
}
