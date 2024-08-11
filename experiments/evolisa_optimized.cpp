#include "io/png.h"
#include "io/image.h"

#include "raster/composit.h"
#include "raster/rasterization.h"

#include <vector>
#include <random>
#include <algorithm>

/* Configuration Parameters */
struct Settings {
    static constexpr int alphaMutationRate = 30;
    static constexpr int alphaRangeMax = 120;
    static constexpr int alphaRangeMin = 30;
    static constexpr int blueMutationRate = 30;
    static constexpr int blueRangeMax = 255;
    static constexpr int blueRangeMin = 0; // missing
    static constexpr int greenMutationRate = 30;
    static constexpr int greenRangeMax = 255;
    static constexpr int greenRangeMin = 0; // missing
    static constexpr int movePointMaxMutationRate = 30;
    static constexpr int movePointMidMutationRate = 30;
    static constexpr int movePointMinMutationRate = 30;
    static constexpr float movePointRangeMid = 20.0 / 200.0; // normalized
    static constexpr float movePointRangeMin = 3.0 / 200.0; // normalized
    static constexpr int pointsMax = 1500;
    static constexpr int pointsMin = 0; // missing
    static constexpr int pointsPerPolygonMax = 10;
    static constexpr int pointsPerPolygonMin = 3;
    static constexpr int polygonsMax = 150;
    static constexpr int polygonsMin = 150; // missing
    static constexpr int redMutationRate = 30;
    static constexpr int redRangeMax = 255;
    static constexpr int redRangeMin = 0; // missing
    static constexpr int removePointMutationRate = 30;
    static constexpr int addPointMutationRate = 30;
    static constexpr int randomizePolygonMutationRate = 30;
};

/* Random Number Utilities */

float randf01() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    return dis(gen);
}

float randf_in_range(float min, float max) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int randint_in_range(int min, int max) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
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
    if (one_in_n(Settings::randomizePolygonMutationRate)) {
        p = random_polygon();
        return true;
    }

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


io::Image<io::RGBA255> render_drawing(const Drawing& d, int width, int height) {
    io::Image<io::RGBA255> image(width, height, io::RGBA255{0, 0, 0, 255});

    for (const auto& polygon : d.polygons) {
        raster::rasterize_polygon_onto_image(polygon.vertices, polygon.colour, image);
    }

    return image;
}

unsigned long long int compute_fitness(const Drawing& d, const io::Image<io::RGBA255>& target) {
    auto image = render_drawing(d, target.width, target.height);
    long long int fitness = 0;

    for (int i = 0; i < target.width * target.height; i++) {
        auto pixel_source = image.data[i];
        auto pixel_target = target.data[i];
        fitness += l2_difference(pixel_source, pixel_target);
        // int delta_r = pixel_source.r - pixel_target.r;
        // int delta_g = pixel_source.g - pixel_target.g;
        // int delta_b = pixel_source.b - pixel_target.b;
        // fitness += (unsigned int)(delta_r * delta_r + delta_g * delta_g + delta_b * delta_b);
    }

    return fitness;
}

bool pixels_equal(const io::RGBA255& a, const io::RGBA255& b) {
    return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

bool images_equal(const io::Image<io::RGBA255>& a, const io::Image<io::RGBA255>& b) {
    if (a.width != b.width || a.height != b.height) {
        return false;
    }

    for (int i = 0; i < a.width * a.height; i++) {
        if (!pixels_equal(a.data[i], b.data[i])) {
            return false;
        }
    }

    return true;
}

int main() {

    io::Image<io::RGBA255> target = io::load_png_rgba("target.png");

    Drawing d = new_drawing();

    int total_iterations = 500;
    // int iterations_per_batch = 1000;

    int total_tick_messages = 50;
    int tick_frequency = total_iterations / total_tick_messages + (total_iterations % total_tick_messages != 0);

    unsigned long long int current_fitness = compute_fitness(d, target);
    double accuracy = 1.0 - (double)current_fitness / (255.0 * 255.0 * 3.0 * target.width * target.height);
    for (int i = 0; i < total_iterations; i++) {

        std::vector<Drawing> candidates(Settings::polygonsMax, d);

        auto image = render_drawing(d, target.width, target.height);

        for (int j = 0; j < Settings::polygonsMax; j++) {
            io::Image<io::RGBA255> background(target.width, target.height, io::RGBA255{0, 0, 0, 255});
            io::Image<io::RGBA255> foreground(target.width, target.height, io::RGBA255{0, 0, 0, 0});

            for (int k = 0; k < Settings::polygonsMax; k++) {
                if (k < j) {
                    raster::rasterize_polygon_onto_image(d.polygons[k].vertices, d.polygons[k].colour, background);
                } else if (k > j) {
                    raster::rasterize_polygon_onto_image(d.polygons[k].vertices, d.polygons[k].colour, foreground);
                }
            }

            long long int baseline_error = 0;
            io::Image<int> error_mask(target.width, target.height, 0);
            for (int x = 0; x < target.width; x++) {
                for (int y = 0; y < target.height; y++) {
                    error_mask.data[x + y * target.width] = l2_difference(raster::composit_over_straight_255(background.data[x + y * target.width], foreground.data[x + y * target.width]), target.data[x + y * target.width]);
                    baseline_error += error_mask.data[x + y * target.width];
                }
            }

            long long int polygon_error = 0;
            for (int k = 0; k < (accuracy < 0.99 ? 10 : 20); k++) {
                Drawing tmp = candidates[j];
                bool dirty = mutate_polygon(tmp, tmp.polygons[j]);
                if (!dirty) continue;

                // auto tmp_test_image = render_drawing(tmp, target.width, target.height);
                // unsigned long long int error = compute_fitness(tmp, target);

                // io::Image<io::RGBA255> test_image3(target.width, target.height, io::RGBA255{0, 0, 0, 255});
                // for (int z = 0; z < Settings::polygonsMax; z++) {
                //     raster::rasterize_polygon_onto_image(tmp.polygons[z].vertices, tmp.polygons[z].colour, test_image3);
                // }

                // assert(images_equal(tmp_test_image, test_image3));

                // io::Image<io::RGBA255> tmp_background(target.width, target.height, io::RGBA255{0, 0, 0, 255});
                // io::Image<io::RGBA255> tmp_polygon(target.width, target.height, io::RGBA255{0, 0, 0, 0});
                // io::Image<io::RGBA255> tmp_foreground(target.width, target.height, io::RGBA255{0, 0, 0, 0});
                // io::Image<io::RGBA255> composited(target.width, target.height, io::RGBA255{0, 0, 0, 0});
                // for (int z = 0; z < Settings::polygonsMax; z++) {
                //     if (z < j) {
                //         raster::rasterize_polygon_onto_image(tmp.polygons[z].vertices, tmp.polygons[z].colour, tmp_background);
                //     } else if (z == j) {
                //         raster::rasterize_polygon_onto_image(tmp.polygons[z].vertices, tmp.polygons[z].colour, tmp_polygon);
                //     } else {
                //         raster::rasterize_polygon_onto_image(tmp.polygons[z].vertices, tmp.polygons[z].colour, tmp_foreground);
                //     }
                // }
                // for (int x = 0; x < target.width; x++) {
                //     for (int y = 0; y < target.height; y++) {
                //         composited.data[x + y * target.width] = raster::composit_over_straight_255(raster::composit_over_straight_255(tmp_background.data[x + y * target.width], tmp_polygon.data[x + y * target.width]), tmp_foreground.data[x + y * target.width]);
                //         // assert (pixels_equal(composited.data[x + y * target.width], tmp_test_image.data[x + y * target.width]));
                //     }
                // }
                // assert(images_equal(composited, tmp_test_image));

                // io::Image<io::RGBA255> test_image = background;
                // raster::rasterize_polygon_onto_image(tmp.polygons[j].vertices, tmp.polygons[j].colour, test_image);
                // long long int total_test_error = 0;
                // io::Image<io::RGBA255> test_image2(target.width, target.height, io::RGBA255{0, 0, 0, 0});
                // for (int x = 0; x < target.width; x++) {
                //     for (int y = 0; y < target.height; y++) {
                //         auto pix = raster::composit_over_straight_255(test_image.data[x + y * target.width], foreground.data[x + y * target.width]);
                //         test_image2.data[x + y * target.width] = pix;
                //         total_test_error += l2_difference(pix, target.data[x + y * target.width]);;
                //     }
                // }

                // assert(images_equal(test_image2, tmp_test_image));

                raster::DrawLossFGShader shader(target, foreground, background, tmp.polygons[j].colour, error_mask);
                raster::rasterize_polygon_scanline(tmp.polygons[j].vertices, target.width, target.height, shader);

                // if (shader.total_error == 0 && error != baseline_error) {
                //     printf("Error mismatch: %lld vs %lld\n", error, baseline_error);
                // }

                // if ((error - current_fitness) != shader.total_error) {
                //     printf("Error mismatch: %lld vs %lld (%lld)\n", error, current_fitness + shader.total_error, shader.total_error);
                // }

                if (shader.total_error < polygon_error) {
                    polygon_error = shader.total_error;
                    candidates[j] = tmp;
                }
            }
        }

        for (int j = 0; j < Settings::polygonsMax; j++) {
            if (compute_fitness(candidates[j], target) < current_fitness) {
                d.polygons[j] = candidates[j].polygons[j];
            }
        }

        current_fitness = compute_fitness(d, target);
        accuracy = 1.0 - (double)current_fitness / (255.0 * 255.0 * 3.0 * target.width * target.height);

        if (i % tick_frequency == 0) {
            printf("Iteration %d/%d | Loss %llu (%.4f%%)\n", i, total_iterations, current_fitness, float(accuracy));
            io::Image<io::RGBA255> result = render_drawing(d, target.width, target.height);
            io::save_png_rgba("result.png", result);
        }
    }

    io::Image<io::RGBA255> result = render_drawing(d, target.width, target.height);
    io::save_png_rgba("result.png", result);
}
