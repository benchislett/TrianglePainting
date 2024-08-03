
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>

int main(int argc, char** argv) {
    int num_triangles = 50;

    if (argc > 1) {
        num_triangles = std::atoi(argv[1]);
    }

    auto json = nlohmann::json();

    // init random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    json["triangles"] = nlohmann::json::array();
    for (int i = 0; i < num_triangles; i++) {
        // random vertices and random colour
        json["triangles"].push_back({
            {"vertices", {
                {dis(gen), dis(gen)},
                {dis(gen), dis(gen)},
                {dis(gen), dis(gen)}
            }},
            {"colour", {dis(gen), dis(gen), dis(gen), 0.5f}}
        });
    }
    json["background"] = {0.0f, 0.0f, 0.0f, 1.0f};

    std::ofstream out("random_triangles.json");
    out << json.dump(4) << std::endl;
}