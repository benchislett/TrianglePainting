#pragma once

#include "common.h"
#include "colours.h"

#include <string>

template<typename PixelT>
struct ImageView {
    using PixelPtr = PixelT*;
    PixelPtr m_data;
    int m_width;
    int m_height;

private:
    void debug_check(int x, int y) const {
        debug_check_ptr(m_data);
        debug_check_range(x, 0, m_width);
        debug_check_range(y, 0, m_height);
    }

    void debug_check(int i) const {
        debug_check_ptr(m_data);
        debug_check_range(i, 0, size());
    }

public:
    int width() const { return m_width; }
    int height() const { return m_height; }

    PixelT* data() { return m_data; }
    const PixelT* data() const { return m_data; }

    PixelT operator()(int x, int y) const {
        debug_check(x, y);
        return m_data[y * m_width + x];
    }

    PixelT& operator()(int x, int y) {
        debug_check(x, y);
        return m_data[y * m_width + x];
    }

    PixelT operator[](int i) const {
        debug_check(i);
        return m_data[i];
    }

    PixelT& operator[](int i) {
        debug_check(i);
        return m_data[i];
    }

    int size() const {
        return m_width * m_height;
    }

    bool empty() const {
        return m_width == 0 || m_height == 0;
    }

    void invalidate() {
        m_data = nullptr;
        m_width = 0;
        m_height = 0;
    }
};

template<typename PixelT>
struct Image : ImageView<PixelT> {
    Image() : ImageView<PixelT>{nullptr, 0, 0} {}

    Image(int width, int height, PixelT initial_value = PixelT()) : ImageView<PixelT>{nullptr, width, height} {
        allocate();
        std::fill(this->data(), this->data() + this->size(), initial_value);
    }

    ~Image() {
        release();
    }

    Image(const Image& other) : ImageView<PixelT>{nullptr, other.width(), other.height()} {
        if (other.data()) {
            allocate();
            std::copy(other.data(), other.data() + other.size(), this->data());
        }
    }

    Image& operator=(const Image& other) {
        if (this != &other) {
            Image tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    Image(Image&& other) noexcept
        : ImageView<PixelT>{other.data(), other.width(), other.height()} {
        other.invalidate();
    }

    Image& operator=(Image&& other) noexcept {
        if (this == &other) return *this;
        release();
        this->m_data = other.data();
        this->m_width = other.width();
        this->m_height = other.height();
        other.invalidate();
        return *this;
    }

    ImageView<PixelT> release_view() noexcept {
        ImageView<PixelT> view = {this->m_data, this->m_width, this->m_height};
        this->invalidate();
        return view;
    }

private:
    void allocate() {
        if (this->size() == 0) {
            this->m_data = nullptr;
            return;
        }
        this->m_data = new PixelT[this->size()];
    }

    void release() {
        delete[] this->m_data;
        this->m_data = nullptr;
    }
};

Image<RGB01> to_rgb01(const ImageView<RGB255>& image);
Image<RGBA01> to_rgba01(const ImageView<RGBA255>& image);
Image<RGB255> to_rgb255(const ImageView<RGB01>& image);
Image<RGBA255> to_rgba255(const ImageView<RGBA01>& image);

Image<RGB255> load_png_rgb(const std::string& filename);
Image<RGBA255> load_png_rgba(const std::string& filename);

void save_png_rgb(const std::string& filename, const ImageView<RGB255>& image);
void save_png_rgba(const std::string& filename, const ImageView<RGBA255>& image);
