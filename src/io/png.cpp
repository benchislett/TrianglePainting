#include "io/png.h"

#include "lodepng.h"

#include <iostream>
#include <algorithm>
#include <cassert>

namespace io {
    LodePNGColorType channels_to_colortype(unsigned int channels) {
        assert (channels > 0 && channels <= 4);
        LodePNGColorType colormodes[4] = {LCT_GREY, LCT_GREY_ALPHA, LCT_RGB, LCT_RGBA};
        return colormodes[channels-1];
    }

    template<typename PixelT>
    static Image<PixelT> load_png_impl(const std::string& filename) {
        using UCharPixelT = std::conditional_t<PixelT::channels == 3, RGB255, RGBA255>;

        
        std::vector<unsigned char> outbuf; // TODO: eliminate copy
        std::vector<unsigned char> buf;
        unsigned int tmp_width, tmp_height;
        unsigned error = lodepng::load_file(buf, filename);
        if(!error) error = lodepng::decode(outbuf, tmp_width, tmp_height, buf, channels_to_colortype(PixelT::channels));
        if(error) std::cerr << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        assert (!error);
        Image<UCharPixelT> out(tmp_width, tmp_height);
        std::copy(outbuf.begin(), outbuf.end(), (unsigned char*)out.data()); // copy here
        return out;
    }

    Image<RGB255> load_png_rgb(const std::string& filename) { return load_png_impl<RGB255>(filename); }
    Image<RGBA255> load_png_rgba(const std::string& filename) { return load_png_impl<RGBA255>(filename); }

    template<typename PixelT>
    static void save_png_impl(const std::string& filename, const ImageView<PixelT>& image) {
        std::vector<unsigned char> png;
        std::vector<unsigned char> buf(image.size() * PixelT::channels); // TODO: eliminate copy
        std::copy((unsigned char*)image.data(), (unsigned char*)image.data() + image.size() * PixelT::channels, buf.begin());
        unsigned error = lodepng::encode(png, buf, image.width(), image.height(), channels_to_colortype(PixelT::channels));
        if(!error) lodepng::save_file(png, filename);
        if(error) std::cerr << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        assert (!error);
    }

    void save_png_rgb(const std::string& filename, const ImageView<RGB255>& image) { save_png_impl(filename, image); }
    void save_png_rgba(const std::string& filename, const ImageView<RGBA255>& image) { save_png_impl(filename, image); }
};
