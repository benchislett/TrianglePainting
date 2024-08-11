#pragma once

#include <vector>

namespace io {
    struct RGB01 {
        constexpr static int channels = 3;
        using data_type = float;

        float r, g, b;
    };

    struct RGBA01 {
        constexpr static int channels = 4;
        using data_type = float;

        float r, g, b, a;
    };

    struct RGB255 {
        constexpr static int channels = 3;
        using data_type = unsigned char;

        unsigned char r, g, b;
    };

    struct RGBA255 {
        constexpr static int channels = 4;
        using data_type = unsigned char;

        unsigned char r, g, b, a;
    };

    RGB01 to_rgb01(const RGB255& rgb);
    RGBA01 to_rgba01(const RGBA255& rgba);
    RGB255 to_rgb255(const RGB01& rgb);
    RGBA255 to_rgba255(const RGBA01& rgba);

    template<typename PixelT>
    struct Image {
        std::vector<PixelT> data;
        int width;
        int height;
    };

    Image<RGB01> to_rgb01(const Image<RGB255>& image);
    Image<RGBA01> to_rgba01(const Image<RGBA255>& image);
    Image<RGB255> to_rgb255(const Image<RGB01>& image);
    Image<RGBA255> to_rgba255(const Image<RGBA01>& image);
};
