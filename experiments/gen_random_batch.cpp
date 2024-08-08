
#include <random>

#include "triangle_image_state.h"

int main(int argc, char** argv) {
    int num_triangles = 50;

    if (argc > 1) {
        num_triangles = std::atoi(argv[1]);
    }

    // init random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    std::vector<geometry2d::triangle> triangles;
    std::vector<io::RGBA255> colours;
    io::RGBA255 background_colour{0, 0, 0, 255};
    for (int i = 0; i < num_triangles; i++) {
        triangles.push_back({
            {dis(gen), dis(gen)},
            {dis(gen), dis(gen)},
            {dis(gen), dis(gen)}
        });
        io::RGBA01 colour = {dis(gen), dis(gen), dis(gen), 0.5f};
        colours.push_back(io::to_rgba255(colour));
    }

    RasterScene scene{triangles, colours, background_colour};
    save_json("random_triangles.json", scene);
}