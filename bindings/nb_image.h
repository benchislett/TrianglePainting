#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "colours.h"
#include "image.h"

#include <cassert>

using PyRGB = nanobind::ndarray<uint8_t, nanobind::shape<3>, nanobind::c_contig, nanobind::device::cpu>;
using PyRGBA = nanobind::ndarray<uint8_t, nanobind::shape<4>, nanobind::c_contig, nanobind::device::cpu>;
using PyRGBFloat = nanobind::ndarray<float, nanobind::shape<3>, nanobind::c_contig, nanobind::device::cpu>;
using PyRGBAFloat = nanobind::ndarray<float, nanobind::shape<4>, nanobind::c_contig, nanobind::device::cpu>;

inline RGB255 depythonize_rgb255(PyRGB rgb) {
    return RGB255(rgb.data()[0], rgb.data()[1], rgb.data()[2]);
}

inline RGBA255 depythonize_rgba255(PyRGBA rgba) {
    return RGBA255(rgba.data()[0], rgba.data()[1], rgba.data()[2], rgba.data()[3]);
}

inline RGB01 depythonize_rgb01(PyRGBFloat rgb) {
    return RGB01(rgb.data()[0], rgb.data()[1], rgb.data()[2]);
}

inline RGBA01 depythonize_rgba01(PyRGBAFloat rgba) {
    return RGBA01(rgba.data()[0], rgba.data()[1], rgba.data()[2], rgba.data()[3]);
}

using PyImageRGB = nanobind::ndarray<uint8_t, nanobind::shape<-1, -1, 3>, nanobind::c_contig, nanobind::device::cpu>;
using PyImageRGBA = nanobind::ndarray<uint8_t, nanobind::shape<-1, -1, 4>, nanobind::c_contig, nanobind::device::cpu>;
using PyImageRGBFloat = nanobind::ndarray<float, nanobind::shape<-1, -1, 3>, nanobind::c_contig, nanobind::device::cpu>;
using PyImageRGBAFloat = nanobind::ndarray<float, nanobind::shape<-1, -1, 4>, nanobind::c_contig, nanobind::device::cpu>;

inline PyImageRGB pythonize_imageview_rgb255(ImageView<RGB255> image) {
    uint8_t* data_ptr = (uint8_t*)(image.data());
    nanobind::capsule owner((void*) data_ptr, [](void* ptr) noexcept { delete[] (RGB255*)(ptr); });
    return PyImageRGB(data_ptr, {(unsigned long)image.width(), (unsigned long)image.height(), 3}, owner);
}

inline PyImageRGBA pythonize_imageview_rgba255(ImageView<RGBA255> image) {
    uint8_t* data_ptr = (uint8_t*)(image.data());
    nanobind::capsule owner((void*) data_ptr, [](void* ptr) noexcept { delete[] (RGBA255*)(ptr); });
    return PyImageRGBA(data_ptr, {(unsigned long)image.width(), (unsigned long)image.height(), 4}, owner);
}

inline PyImageRGBFloat pythonize_imageview_rgb01(ImageView<RGB01> image) {
    float* data_ptr = (float*)(image.data());
    nanobind::capsule owner((void*) data_ptr, [](void* ptr) noexcept { delete[] (RGB01*)(ptr); });
    return PyImageRGBFloat(data_ptr, {(unsigned long)image.width(), (unsigned long)image.height(), 3}, owner);
}

inline PyImageRGBAFloat pythonize_imageview_rgba01(ImageView<RGBA01> image) {
    float* data_ptr = (float*)(image.data());
    nanobind::capsule owner((void*) data_ptr, [](void* ptr) noexcept { delete[] (RGBA01*)(ptr); });
    return PyImageRGBAFloat(data_ptr, {(unsigned long)image.width(), (unsigned long)image.height(), 4}, owner);
}

inline ImageView<RGB255> depythonize_imageview_rgb255(PyImageRGB image) {
    assert(image.size() == 3);
    assert(image.shape(2) == 3);
    return ImageView<RGB255>((RGB255*)(image.data()), image.shape(0), image.shape(1));
}

inline ImageView<RGBA255> depythonize_imageview_rgba255(PyImageRGBA image) {
    assert(image.size() == 3);
    assert(image.shape(2) == 4);
    return ImageView<RGBA255>((RGBA255*)(image.data()), image.shape(0), image.shape(1));
}

inline ImageView<RGB01> depythonize_imageview_rgb01(PyImageRGBFloat image) {
    assert(image.size() == 3);
    assert(image.shape(2) == 3);
    return ImageView<RGB01>((RGB01*)(image.data()), image.shape(0), image.shape(1));
}

inline ImageView<RGBA01> depythonize_imageview_rgba01(PyImageRGBAFloat image) {
    assert(image.size() == 3);
    assert(image.shape(2) == 4);
    return ImageView<RGBA01>((RGBA01*)(image.data()), image.shape(0), image.shape(1));
}
