#pragma once

#include "io/image.h"
#include "geometry/types.h"

#include <nlohmann/json.hpp>

#include <vector>
#include <fstream>

struct RasterScene {
    std::vector<geometry2d::triangle> triangles;
    std::vector<io::RGBA255> colours;
    io::RGBA255 background_colour;
};

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
    return RasterScene{triangles, colours, io::to_rgba255(background)};
}

void save_json(const std::string& filename, const RasterScene& scene) {
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
