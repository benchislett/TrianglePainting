#include <iostream>
#include <chrono>
#include <vector>
#include "../include/rasterize.h"
#include "../include/shaders.h"
#include "../include/image.h"
#include "../include/colours.h"

struct DummyShader {
    ImageView<RGBA255> background;
    RGBA255 colour;

    DummyShader(const ImageView<RGBA255>& bg, const RGBA255& col) : background(bg), colour(col) {}
    
    void render_pixel(int x, int y) {
        background(x, y) = colour;
    }
};

int main() {
    constexpr int image_size = 500;
    Image<RGBA255> background_img(image_size, image_size);
    ImageView<RGBA255> background_view(background_img);
    std::fill(background_img.begin(), background_img.end(), RGBA255{0, 0, 0, 255});

    CompositOverShader shader(background_view, RGBA255{255, 0, 0, 128});
    // struct Shader {};
    // DummyShader shader(background_view, RGBA255{255, 0, 0, 128});

    auto shape = std::make_shared<Triangle>(
        Triangle{}
    );
    shape->vertices = std::array<Point, 3>{Point{0.2f, 0.2f}, Point{0.8f, 0.2f}, Point{0.5f, 0.8f}};

    std::vector<RasterStrategy> strategies = {
        RasterStrategy::Bounded,
        RasterStrategy::Integer,
        RasterStrategy::ScanlinePolygon,
        RasterStrategy::TestNewPolygon
    };

    for (auto strategy : strategies) {
        RasterConfig config;
        config.strategy = strategy;
        config.image_width = image_size;
        config.image_height = image_size;

        const int N = 10000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            std::fill(background_img.begin(), background_img.end(), RGBA255{0, 0, 0, 255});
            rasterize(shape, shader, config);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        double ips = N / duration.count();

        std::cout << "Strategy=" << static_cast<int>(strategy)
                  << " Iterations/s=" << ips << std::endl;
    }
    return 0;
}
