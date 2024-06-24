#pragma once

#include "lodepng.h"

#include <vector>

namespace io {

    /* Generic image data object, where `channels` can be 1 (grayscale), 2 (greyscale + alpha), 3 (RGB), or 4 (RGBA). */
    struct Image {
        std::vector<unsigned char> data;
        unsigned int width;
        unsigned int height;
        unsigned int channels;
    };

    Image load_png(const std::string& filename, unsigned int channels = 4);
    void save_png(const std::string& filename, const Image& image);
};
