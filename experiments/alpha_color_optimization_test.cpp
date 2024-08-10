#include "io/png.h"
#include "io/image.h"

#include "raster/composit.h"
#include "raster/rasterization.h"

#include <vector>
#include <random>
#include <algorithm>


template<typename T>
io::Image<T> blank_image(int width, int height, T value) {
    io::Image<T> image;
    image.width = width;
    image.height = height;
    image.data.resize(width * height);
    std::fill(image.data.begin(), image.data.end(), value);
    return image;
}

long long int image_delta(const io::Image<io::RGBA255>& a, const io::Image<io::RGBA255>& b) {
    long long int delta = 0;
    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            int delta_r = a.data[i * a.width + j].r - b.data[i * b.width + j].r;
            int delta_g = a.data[i * a.width + j].g - b.data[i * b.width + j].g;
            int delta_b = a.data[i * a.width + j].b - b.data[i * b.width + j].b;
            delta += delta_r * delta_r + delta_g * delta_g + delta_b * delta_b;
        }
    }
    return delta;
}

io::Image<int> image_delta_mask(const io::Image<io::RGBA255>& a, const io::Image<io::RGBA255>& b) {
    auto mask = blank_image<int>(a.width, a.height, 0);
    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            int delta_r = a.data[i * a.width + j].r - b.data[i * b.width + j].r;
            int delta_g = a.data[i * a.width + j].g - b.data[i * b.width + j].g;
            int delta_b = a.data[i * a.width + j].b - b.data[i * b.width + j].b;
            mask.data[i * a.width + j] = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b;
        }
    }
    return mask;
}

int main() {
    int width = 1;
    io::Image<io::RGBA255> background = blank_image(width, width, io::RGBA255{0, 0, 0, 255});
    io::Image<io::RGBA255> foreground = blank_image(width, width, io::RGBA255{0, 0, 0, 128});

    io::Image<io::RGBA255> target = blank_image(width, width, io::RGBA255{192, 0, 0, 255});

    unsigned char alpha = 94;

    auto initial_colour = io::RGBA255{100, 100, 50, alpha};
    auto initial_composited_pixel = raster::composit_over_straight_255(raster::composit_over_straight_255(background.data[0], initial_colour), foreground.data[0]);

    long long int initial_error = image_delta(target, blank_image(width, width, initial_composited_pixel));
    auto initial_error_mask = image_delta_mask(target, blank_image(width, width, initial_composited_pixel));

    raster::OptimalColourShader shader(alpha, target, foreground, background, initial_error_mask);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            shader.render_pixel(i, j);
        }
    }

    auto [final_colour, error_delta] = shader.final_colour_and_error();
    auto final_composited_pixel = raster::composit_over_straight_255(raster::composit_over_straight_255(background.data[0], final_colour), foreground.data[0]);

    long long int new_error = image_delta(target, blank_image(width, width, final_composited_pixel));
    auto new_error_mask = image_delta_mask(target, blank_image(width, width, final_composited_pixel));

    printf("Initial error: %lld\n", initial_error);
    printf("Final error: %lld | Estimated at %lld (Delta %lld)\n", new_error, initial_error + error_delta, error_delta);
    printf("Final colour: %d %d %d | Composited to %d %d %d (Alpha %d)\n", int(final_colour.r), int(final_colour.g), int(final_colour.b), int(final_composited_pixel.r), int(final_composited_pixel.g), int(final_composited_pixel.b), int(final_composited_pixel.a));

    long long int delta_discrepancy = abs(new_error - (initial_error + error_delta));
    if (delta_discrepancy > 0) {
        printf("Error, discrepancy of %lld\n", delta_discrepancy);
    }

    return 0;
}
