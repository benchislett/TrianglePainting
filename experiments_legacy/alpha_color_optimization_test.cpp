#include "io/png.h"
#include "io/image.h"

#include "raster/composit.h"
#include "raster/rasterization.h"

#include <vector>
#include <random>
#include <algorithm>

long long int image_delta(const io::ImageView<io::RGBA255>& a, const io::ImageView<io::RGBA255>& b) {
    long long int delta = 0;
    for (int i = 0; i < a.size(); i++) {
        int delta_r = a[i].r - b[i].r;
        int delta_g = a[i].g - b[i].g;
        int delta_b = a[i].b - b[i].b;
        delta += delta_r * delta_r + delta_g * delta_g + delta_b * delta_b;
    }
    return delta;
}

io::Image<int> image_delta_mask(const io::ImageView<io::RGBA255>& a, const io::ImageView<io::RGBA255>& b) {
    auto mask = io::Image<int>(a.width(), a.height(), 0);
    for (int i = 0; i < a.size(); i++) {
        int delta_r = a[i].r - b[i].r;
        int delta_g = a[i].g - b[i].g;
        int delta_b = a[i].b - b[i].b;
        mask[i] = delta_r * delta_r + delta_g * delta_g + delta_b * delta_b;
    }
    return mask;
}

int main() {
    io::Image<io::RGBA255> target = io::load_png_rgba("target.png");

    int width = target.width();
    assert (width == target.height());

    io::Image<io::RGBA255> background = io::Image<io::RGBA255>(width, width, io::RGBA255{0, 0, 16, 255});
    io::Image<io::RGBA255> foreground = io::Image<io::RGBA255>(width, width, io::RGBA255{0, 0, 130, 16});

    unsigned char alpha = 16;

    auto initial_colour = io::RGBA255{100, 100, 50, alpha};
    auto initial_composited_pixel = raster::composit_over_straight_255(raster::composit_over_straight_255(background[0], initial_colour), foreground[0]);

    long long int initial_error = image_delta(target, io::Image<io::RGBA255>(width, width, initial_composited_pixel));
    auto initial_error_mask = image_delta_mask(target, io::Image<io::RGBA255>(width, width, initial_composited_pixel));

    raster::OptimalColourShader shader(alpha, target, foreground, background, initial_error_mask);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            shader.render_pixel(i, j);
        }
    }

    auto [final_colour, error_delta] = shader.final_colour_and_error();
    auto final_composited_pixel = raster::composit_over_straight_255(raster::composit_over_straight_255(background[0], final_colour), foreground[0]);

    long long int new_error = image_delta(target, io::Image<io::RGBA255>(width, width, final_composited_pixel));
    auto new_error_mask = image_delta_mask(target, io::Image<io::RGBA255>(width, width, final_composited_pixel));

    printf("Initial error: %lld\n", initial_error);
    printf("Final error: %lld | Estimated at %lld (Delta %lld)\n", new_error, initial_error + error_delta, error_delta);
    printf("Final colour: %d %d %d | Composited to %d %d %d (Alpha %d)\n", int(final_colour.r), int(final_colour.g), int(final_colour.b), int(final_composited_pixel.r), int(final_composited_pixel.g), int(final_composited_pixel.b), int(final_composited_pixel.a));

    long long int delta_discrepancy = abs(new_error - (initial_error + error_delta));
    if (delta_discrepancy > 0) {
        printf("Error, discrepancy of %lld\n", delta_discrepancy);
    }

    io::RGBA255 test_colour = final_colour;
    for (int b = 0; b < 255; b++) {
        test_colour.b = (unsigned char)b;
        auto test_composited_pixel = raster::composit_over_straight_255(raster::composit_over_straight_255(background[0], test_colour), foreground[0]);
        long long int test_error = image_delta(target, io::Image<io::RGBA255>(width, width, test_composited_pixel));
        printf("%lld,", test_error);
    }
    printf("\n");

    return 0;
}
