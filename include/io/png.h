#pragma once

#include "lodepng.h"

#include "io/image.h"

#include <string>

namespace io {

    Image<RGB255> load_png_rgb(const std::string& filename);
    Image<RGBA255> load_png_rgba(const std::string& filename);

    void save_png_rgb(const std::string& filename, const ImageView<RGB255>& image);
    void save_png_rgba(const std::string& filename, const ImageView<RGBA255>& image);
};
