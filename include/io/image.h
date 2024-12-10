#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <utility>

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

    unsigned int l2_difference(const RGB255& a, const RGB255& b);
    unsigned int l2_difference(const RGBA255& a, const RGBA255& b);
    unsigned int l1_difference(const RGB255& a, const RGB255& b);
    unsigned int l1_difference(const RGBA255& a, const RGBA255& b);

    float l2_difference(const RGB01& a, const RGB01& b);
    float l2_difference(const RGBA01& a, const RGBA01& b);
    float l1_difference(const RGB01& a, const RGB01& b);
    float l1_difference(const RGBA01& a, const RGBA01& b);

    RGB01 to_rgb01(const RGB255& rgb);
    RGBA01 to_rgba01(const RGBA255& rgba);
    RGB255 to_rgb255(const RGB01& rgb);
    RGBA255 to_rgba255(const RGBA01& rgba);

    template<typename PixelT>
    struct ImageView {
        using PixelPtr = PixelT*;
        PixelPtr m_data;
        int m_width;
        int m_height;
    
    private:
        void debug_check(int x, int y) const {
#ifdef NDEBUG
            (void)x;
            (void)y;
#else
            assert(m_data && x >= 0 && x < m_width && y >= 0 && y < m_height);
#endif
        }

        void debug_check(int i) const {
#ifdef NDEBUG
            (void)i;
#else
            assert(m_data && i >= 0 && i < size());
#endif
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
};
