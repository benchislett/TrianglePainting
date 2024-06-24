#include "io/image.h"

#include <iostream>
#include <cassert>

namespace io {

    LodePNGColorType channels_to_colortype(unsigned int channels) {
        assert (channels > 0 && channels <= 4);
        LodePNGColorType colormodes[4] = {LCT_GREY, LCT_GREY_ALPHA, LCT_RGB, LCT_RGBA};
        return colormodes[channels-1];
    }

    Image load_png(const std::string& filename, unsigned int channels) {
        Image out;
        std::vector<unsigned char> buf;
        unsigned error = lodepng::load_file(buf, filename);
        if(!error) error = lodepng::decode(out.data, out.width, out.height, buf, channels_to_colortype(channels));
        if(error) std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        out.channels = 4;
        return out;
    }

    void save_png(const std::string& filename, const Image& image) {
        std::vector<unsigned char> png;
        unsigned error = lodepng::encode(png, image.data, image.width, image.height, channels_to_colortype(image.channels));
        if(!error) lodepng::save_file(png, filename);
        if(error) std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
    }
};