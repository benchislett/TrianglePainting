#include "io/image.h"

namespace io {
    RGB01 to_rgb01(const RGB255 &rgb)
    {
        return RGB01{(float)rgb.r / 255, (float)rgb.g / 255, (float)rgb.b / 255};
    }

    RGBA01 to_rgba01(const RGBA255 &rgba)
    {
        return RGBA01{(RGBA01::data_type)rgba.r / 255, (float)rgba.g / 255, (float)rgba.b / 255, (float)rgba.a / 255};
    }

    RGB255 to_rgb255(const RGB01 &rgb)
    {
        return RGB255{(unsigned char)(rgb.r * 255.99), (unsigned char)(rgb.g * 255.99), (unsigned char)(rgb.b * 255.99)};
    }

    RGBA255 to_rgba255(const RGBA01 &rgba)
    {
        return RGBA255{(unsigned char)(rgba.r * 255.99), (unsigned char)(rgba.g * 255.99), (unsigned char)(rgba.b * 255.99), (unsigned char)(rgba.a * 255.99)};
    }

    Image<RGB01> to_rgb01(const Image<RGB255> &image)
    {
        Image<RGB01> out;
        out.width = image.width;
        out.height = image.height;
        out.data.reserve(image.data.size());
        for (auto &pixel : image.data)
        {
            out.data.push_back(to_rgb01(pixel));
        }
        return out;
    };

    Image<RGBA01> to_rgba01(const Image<RGBA255> &image)
    {
        Image<RGBA01> out;
        out.width = image.width;
        out.height = image.height;
        out.data.reserve(image.data.size());
        for (auto &pixel : image.data)
        {
            out.data.push_back(to_rgba01(pixel));
        }
        return out;
    };

    Image<RGB255> to_rgb255(const Image<RGB01> &image)
    {
        Image<RGB255> out;
        out.width = image.width;
        out.height = image.height;
        out.data.reserve(image.data.size());
        for (auto &pixel : image.data)
        {
            out.data.push_back(to_rgb255(pixel));
        }
        return out;
    };

    Image<RGBA255> to_rgba255(const Image<RGBA01> &image)
    {
        Image<RGBA255> out;
        out.width = image.width;
        out.height = image.height;
        out.data.reserve(image.data.size());
        for (auto &pixel : image.data)
        {
            out.data.push_back(to_rgba255(pixel));
        }
        return out;
    };
}
