#include "io/png.h"
#include "io/image.h"

#include "raster/composit.h"
#include "raster/rasterization.h"

#include <vector>
#include <random>
#include <algorithm>

/* Configuration Parameters */
struct Settings {
    static constexpr int addPolygonMutationRate = 700;
    static constexpr int alphaMutationRate = 1500;
    static constexpr int alphaRangeMax = 120;
    static constexpr int alphaRangeMin = 30;
    static constexpr int blueMutationRate = 1500;
    static constexpr int blueRangeMax = 255;
    static constexpr int blueRangeMin = 0; // missing
    static constexpr int greenMutationRate = 1500;
    static constexpr int greenRangeMax = 255;
    static constexpr int greenRangeMin = 0; // missing
    static constexpr int movePointMaxMutationRate = 1500;
    static constexpr int movePointMidMutationRate = 1500;
    static constexpr int movePointMinMutationRate = 1500;
    static constexpr float movePointRangeMid = 20.0 / 200.0; // normalized
    static constexpr float movePointRangeMin = 3.0 / 200.0; // normalized
    static constexpr int movePolygonMutationRate = 700;
    static constexpr int pointsMax = 1500;
    static constexpr int pointsMin = 0; // missing
    static constexpr int pointsPerPolygonMax = 10;
    static constexpr int pointsPerPolygonMin = 3;
    static constexpr int polygonsMax = 150;
    static constexpr int polygonsMin = 0; // missing
    static constexpr int redMutationRate = 1500;
    static constexpr int redRangeMax = 255;
    static constexpr int redRangeMin = 0; // missing
    static constexpr int removePointMutationRate = 1500;
    static constexpr int removePolygonMutationRate = 1500;
    static constexpr int addPointMutationRate = 1500;
};

/* Random Number Utilities */

float randf01() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0, 1.0);
    return dis(gen);
}

float randf_in_range(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int randint_in_range(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

bool one_in_n(int n) {
    return randint_in_range(0, n - 1) == 0;
}

/* Point Utilities */

auto random_point() {
    return geometry2d::point{randf01(), randf01()};
}

bool mutate_point(geometry2d::point& p) {
    if (one_in_n(Settings::movePointMaxMutationRate)) {
        p = random_point();
    } else if (one_in_n(Settings::movePointMidMutationRate)) {
        p.x = std::min(1.0f, std::max(0.0f, p.x + randf_in_range(-Settings::movePointRangeMid, Settings::movePointRangeMid)));
        p.y = std::min(1.0f, std::max(0.0f, p.y + randf_in_range(-Settings::movePointRangeMid, Settings::movePointRangeMid)));
    } else if (one_in_n(Settings::movePointMinMutationRate)) {
        p.x = std::min(1.0f, std::max(0.0f, p.x + randf_in_range(-Settings::movePointRangeMin, Settings::movePointRangeMin)));
        p.y = std::min(1.0f, std::max(0.0f, p.y + randf_in_range(-Settings::movePointRangeMin, Settings::movePointRangeMin)));
    } else {
        return false;
    }
    return true;
}

/* Colour Utilities */

auto random_colour() {
    return io::RGBA255{(unsigned char)randint_in_range(Settings::redRangeMin, Settings::redRangeMax),
                       (unsigned char)randint_in_range(Settings::greenRangeMin, Settings::greenRangeMax),
                       (unsigned char)randint_in_range(Settings::blueRangeMin, Settings::blueRangeMax),
                       (unsigned char)randint_in_range(Settings::alphaRangeMin, Settings::alphaRangeMax)};
}

bool mutate_colour(io::RGBA255& c) {
    bool mutated = false;
    if (one_in_n(Settings::redMutationRate)) {
        c.r = (unsigned char)randint_in_range(Settings::redRangeMin, Settings::redRangeMax);
        mutated = true;
    }
    if (one_in_n(Settings::greenMutationRate)) {
        c.g = (unsigned char)randint_in_range(Settings::greenRangeMin, Settings::greenRangeMax);
        mutated = true;
    }
    if (one_in_n(Settings::blueMutationRate)) {
        c.b = (unsigned char)randint_in_range(Settings::blueRangeMin, Settings::blueRangeMax);
        mutated = true;
    }
    if (one_in_n(Settings::alphaMutationRate)) {
        c.a = (unsigned char)randint_in_range(Settings::alphaRangeMin, Settings::alphaRangeMax);
        mutated = true;
    }
    return mutated;
}

/* Polygon Utilities */

struct Polygon {
    std::vector<geometry2d::point> vertices;
    io::RGBA255 colour;
};

struct Drawing {
    std::vector<Polygon> polygons;
};

int count_points(const Drawing& d) {
    int count = 0;
    for (auto p : d.polygons) {
        count += p.vertices.size();
    }
    return count;
}

auto random_polygon() {
    Polygon p;
    int num_points = Settings::pointsPerPolygonMin;
    auto origin = random_point();
    p.vertices.push_back(origin);
    for (int i = 1; i < num_points; i++) {
        auto p2 = origin;
        p2.x = std::min(1.0f, std::max(0.0f, p2.x + randf_in_range(-Settings::movePointRangeMin, Settings::movePointRangeMin)));
        p2.y = std::min(1.0f, std::max(0.0f, p2.y + randf_in_range(-Settings::movePointRangeMin, Settings::movePointRangeMin)));
        p.vertices.push_back(p2);
    }

    p.colour = random_colour();
    return p;
}

bool remove_point(const Drawing& d, Polygon& p) {
    if (p.vertices.size() > Settings::pointsPerPolygonMin && count_points(d) > Settings::pointsMin) {
        p.vertices.erase(p.vertices.begin() + randint_in_range(0, p.vertices.size() - 1));
        return true;
    }
    return false;
}

bool add_point(const Drawing& d, Polygon& p) {
    if (p.vertices.size() < Settings::pointsPerPolygonMax && count_points(d) < Settings::pointsMax) {
        int idx = randint_in_range(1, p.vertices.size() - 1);
        const auto& prev = p.vertices[idx - 1];
        const auto& curr = p.vertices[idx];

        auto new_point = geometry2d::point{(prev.x + curr.x) / 2.0f, (prev.y + curr.y) / 2.0f};
        p.vertices.insert(p.vertices.begin() + idx, new_point);
        return true;
    }
    return false;
}

bool mutate_polygon(const Drawing& d, Polygon& p) {
    bool mutated = false;
    if (one_in_n(Settings::addPointMutationRate)) {
        mutated = mutated || add_point(d, p);
    }
    if (one_in_n(Settings::removePointMutationRate)) {
        mutated = mutated || remove_point(d, p);
    }
    mutated = mutated || mutate_colour(p.colour);
    for (auto& point : p.vertices) {
        mutated = mutated || mutate_point(point);
    }
    return mutated;
}

/* Simulation Utilities */

bool add_polygon(Drawing& d) {
    if (d.polygons.size() < Settings::polygonsMax) {
        auto p = random_polygon();
        int position = randint_in_range(0, d.polygons.size());
        d.polygons.insert(d.polygons.begin() + position, p);
        return true;
    }
    return false;
}

bool remove_polygon(Drawing& d) {
    if (d.polygons.size() > Settings::polygonsMin) {
        int index = randint_in_range(0, d.polygons.size() - 1);
        d.polygons.erase(d.polygons.begin() + index);
        return true;
    }
    return false;
}

bool move_polygon(Drawing& d) {
    if (d.polygons.size() > 1) {
        int idx_1 = randint_in_range(0, d.polygons.size() - 1);
        auto polygon = d.polygons[idx_1];
        d.polygons.erase(d.polygons.begin() + idx_1);
        int idx_dest = randint_in_range(0, d.polygons.size()); // note that the size has changed due to the erasure
        d.polygons.insert(d.polygons.begin() + idx_dest, polygon);
        return true;
    }
    return false;
}

auto new_drawing() {
    Drawing d;
    for (int i = 0; i < Settings::polygonsMin; i++) {
        add_polygon(d);
    }
    return d;
}

bool mutate_drawing(Drawing& d) {
    bool mutated = false;
    if (one_in_n(Settings::addPolygonMutationRate)) {
        mutated = mutated || add_polygon(d);
    }
    if (one_in_n(Settings::removePolygonMutationRate)) {
        mutated = mutated || remove_polygon(d);
    }
    if (one_in_n(Settings::movePolygonMutationRate)) {
        mutated = mutated || move_polygon(d);
    }
    for (auto& p : d.polygons) {
        mutated = mutated || mutate_polygon(d, p);
    }
    return mutated;
}

io::Image<io::RGBA255> render_drawing(const Drawing& d, int width, int height) {
    io::Image<io::RGBA255> image;
    image.width = width;
    image.height = height;
    image.data.resize(width * height);
    std::fill(image.data.begin(), image.data.end(), io::RGBA255{0, 0, 0, 255});

    for (const auto& polygon : d.polygons) {
        raster::rasterize_polygon_onto_image(polygon.vertices, polygon.colour, image);
    }

    return image;
}

unsigned long long int compute_fitness(const Drawing& d, const io::Image<io::RGB255>& target) {
    auto image = render_drawing(d, target.width, target.height);
    unsigned long long int fitness = 0;

    for (int i = 0; i < target.width * target.height; i++) {
        auto pixel_source = image.data[i];
        auto pixel_target = target.data[i];
        int delta_r = pixel_source.r - pixel_target.r;
        int delta_g = pixel_source.g - pixel_target.g;
        int delta_b = pixel_source.b - pixel_target.b;
        fitness += (unsigned int)(delta_r * delta_r + delta_g * delta_g + delta_b * delta_b);
    }

    return fitness;
}

int main() {

    io::Image<io::RGB255> target = io::load_png_rgb("target.png");

    Drawing d = new_drawing();

    unsigned long long int current_fitness = compute_fitness(d, target);
    int total_iterations = 10000000;
    int accepted_mutations = 0;

    int total_tick_messages = 100;
    int tick_frequency = total_iterations / total_tick_messages + (total_iterations % total_tick_messages != 0);

    for (int i = 0; i < total_iterations; i++) {
        Drawing d2 = d;
        bool dirty = mutate_drawing(d2);
        if (dirty) {
            unsigned long long int new_fitness = compute_fitness(d2, target);
            if (new_fitness < current_fitness) {
                d = d2;
                current_fitness = new_fitness;
                accepted_mutations++;
            }
        }

        if (i % tick_frequency == 0) {
            double accuracy = 1.0 - (double)current_fitness / (255.0 * 255.0 * 3.0 * target.width * target.height);
            printf("Iteration %d/%d | Loss %llu (%.4f%%)\n", i, total_iterations, current_fitness, float(accuracy));
            io::Image<io::RGBA255> result = render_drawing(d, target.width, target.height);
            io::save_png_rgba("result.png", result);
        }
    }

    printf("Accepted iterations: %d (%.2f%%)\n", accepted_mutations, (float)accepted_mutations / total_iterations * 100.0f);

    io::Image<io::RGBA255> result = render_drawing(d, target.width, target.height);
    io::save_png_rgba("result.png", result);
}
