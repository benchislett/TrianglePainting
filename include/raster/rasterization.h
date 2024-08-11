#pragma once

#include "geometry/types.h"
#include "geometry/barycentric.h"
#include "io/image.h"
#include "raster/composit.h"

#include <vector>
#include <cassert>

#include <unordered_set>

namespace raster {

    /* A full but less-configurable 2D triangle rasterization reference using a hidden OpenGL window. 
     * Takes a list of triangles and respective colours and a background colour, and rasterizes them in a single draw call to the output image.
     * Overwrites any contents of the output image.
     */
    void rasterize_triangles_to_image_opengl(const std::vector<geometry2d::triangle>& triangles, const std::vector<io::RGBA255>& colours, io::RGBA255 background_colour, io::Image<io::RGBA255>& image);

    /* Generic 2D Triangle Rasterization 
     * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `triangle`
     * assuming an image domain of size `width` x `height`. 
     * 
     * Method: Bounded loop.
     * - Determine the integer bounding box of the triangle within the image
     * - For each pixel in the box, compute the barycentric coordinates at the center of that pixel w.r.t. the `triangle`
     * - If the barycentric coordinates are all non-negative, shade the point. */
    template<class Shader>
    void rasterize_triangle_bounded(const geometry2d::triangle& triangle, int width, int height, Shader& shader) {
        int lower_x = std::max(0, (int)(std::min({triangle.a.x, triangle.b.x, triangle.c.x}) * width));
        int lower_y = std::max(0, (int)(std::min({triangle.a.y, triangle.b.y, triangle.c.y}) * height));
        int upper_x = std::min(width - 1, (int)(std::max({triangle.a.x, triangle.b.x, triangle.c.x}) * width));
        int upper_y = std::min(height - 1, (int)(std::max({triangle.a.y, triangle.b.y, triangle.c.y}) * height));

        for (int x = lower_x; x <= upper_x; x++) {
            for (int y = lower_y; y <= upper_y; y++) {
                float u = (x + 0.5f) / (float)width;
                float v = (y + 0.5f) / (float)height;
                auto bary = geometry2d::barycentric_coordinates({u, v}, triangle);
                if (bary.u >= 0 && bary.v >= 0 && bary.w >= 0) {
                    shader.render_pixel(x, y);
                }
            }
        }
    }
      
    /* Generic 2D Triangle Rasterization 
     * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `triangle`
     * assuming an image domain of size `width` x `height`. 
     * 
     * Method: Integer loop.
     * An optimized variant of the bounded loop method that uses exclusively integer arithmetic
     * to compute inclusion by integer edge functions whose values are updated incrementally.
     * 
     * Credit for the implementation goes to https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer
     */
    template<class Shader>
    void rasterize_triangle_integer(const geometry2d::triangle& tri, int width, int height, Shader& shader) {
        int xs[3] = {int(tri.a.x * width), int(tri.b.x * width), int(tri.c.x * width)};
        int ys[3] = {int(tri.a.y * height), int(tri.b.y * height), int(tri.c.y * height)};

        // Orient the triangle correctly
        int w = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]);
        if (w < 0) {
            std::swap(xs[0], xs[1]);
            std::swap(ys[0], ys[1]);
        }

        auto orient2d = [](int ax, int ay, int bx, int by, int cx, int cy) {
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        };

        int lower_x = std::max(0, std::min({xs[0], xs[1], xs[2]}));
        int lower_y = std::max(0, std::min({ys[0], ys[1], ys[2]}));
        int upper_x = std::min(width - 1, std::max({xs[0], xs[1], xs[2]}));
        int upper_y = std::min(height - 1, std::max({ys[0], ys[1], ys[2]}));

        int w0_row = orient2d(xs[1], ys[1], xs[2], ys[2], lower_x, lower_y);
        int w1_row = orient2d(xs[2], ys[2], xs[0], ys[0], lower_x, lower_y);
        int w2_row = orient2d(xs[0], ys[0], xs[1], ys[1], lower_x, lower_y);

        int A01 = ys[0] - ys[1], B01 = xs[1] - xs[0];
        int A12 = ys[1] - ys[2], B12 = xs[2] - xs[1];
        int A20 = ys[2] - ys[0], B20 = xs[0] - xs[2];

        // Rasterize
        for (int y = lower_y; y <= upper_y; y++) {
            int w0 = w0_row;
            int w1 = w1_row;
            int w2 = w2_row;

            for (int x = lower_x; x <= upper_x; x++) {
                // If p is on or inside all edges, render pixel.
                if ((w0 | w1 | w2) >= 0) {
                    shader.render_pixel(x, y);
                }

                // One step to the right
                w0 += A12;
                w1 += A20;
                w2 += A01;
            }

            // One row step
            w0_row += B12;
            w1_row += B20;
            w2_row += B01;
        }
    }

    /* Generic 2D Polygon Rasterization 
     * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `polygon`
     * assuming an image domain of size `width` x `height`. 
     * 
     * Method: Sorted scanlines.
     * - For each row of the image, build a list of nodes where the polygon intersects the row
     * - Sort the nodes by x-coordinate
     * - Fill the pixels between even-odd pairs of nodes
     * 
     * Credit for the implementation goes to http://alienryderflex.com/polygon_fill/
     */
    template<class Shader>
    void rasterize_polygon_scanline(const std::vector<geometry2d::point>& polygon, int width, int height, Shader& shader) {
        constexpr int MAX_POLY_CORNERS = 10;

        int  nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap ;

        int polyCorners = polygon.size();

        //  Loop through the rows of the image.
        for (pixelY=0; pixelY<height; pixelY++) {

            //  Build a list of nodes.
            nodes=0; j=polyCorners-1;
            for (i=0; i<polyCorners; i++) {
                float p_i_x = polygon[i].x * width;
                float p_j_x = polygon[j].x * width;
                float p_i_y = polygon[i].y * height;
                float p_j_y = polygon[j].y * height;
                if (p_i_y<(float) pixelY && p_j_y>=(float) pixelY
                ||  p_j_y<(float) pixelY && p_i_y>=(float) pixelY) {
                nodeX[nodes++]=(int) (p_i_x+(pixelY-p_i_y)/(p_j_y-p_i_y)
                *(p_j_x-p_i_x)); }
                j=i; }

            //  Sort the nodes, via a simple “Bubble” sort.
            i=0;
            while (i<nodes-1) {
                if (nodeX[i]>nodeX[i+1]) {
                swap=nodeX[i]; nodeX[i]=nodeX[i+1]; nodeX[i+1]=swap; if (i) i--; }
                else {
                i++; }}

            //  Fill the pixels between node pairs.
            for (i=0; i<nodes; i+=2) {
                if   (nodeX[i  ]>=width) break;
                if   (nodeX[i+1]> 0 ) {
                if (nodeX[i  ]< 0 ) nodeX[i  ]=0 ;
                if (nodeX[i+1]> width) nodeX[i+1]=width;
                for (pixelX=nodeX[i]; pixelX<nodeX[i+1]; pixelX++) shader.render_pixel(pixelX,pixelY); }}}
    }

    /* Modes for performing 2D triangle rasterization
     * - Bounded: see `rasterize_triangle_bounded`
     * - Integer: see `rasterize_triangle_integer`
     * - ScanlinePolygon: see `rasterize_polygon_scanline`
     */
    enum class TriangleRasterizationMode {
        Bounded,
        Integer,
        ScanlinePolygon,
        Default = Bounded
    };

    /* Generic 2D Triangle Rasterization 
     * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `triangle`
     * assuming an image domain of size `width` x `height`.
     * See: `TriangleRasterizationMode` for available modes.
     */
    template<class Shader, TriangleRasterizationMode mode = TriangleRasterizationMode::Default>
    void rasterize_triangle(const geometry2d::triangle& triangle, int width, int height, Shader& shader) {
        if constexpr (mode == TriangleRasterizationMode::Bounded) {
            rasterize_triangle_bounded(triangle, width, height, shader);
        } else if constexpr (mode == TriangleRasterizationMode::Integer) {
            rasterize_triangle_integer(triangle, width, height, shader);
        } else if constexpr (mode == TriangleRasterizationMode::ScanlinePolygon) {
            std::vector<geometry2d::point> points = {triangle.a, triangle.b, triangle.c};
            rasterize_polygon_scanline(points, width, height, shader);
        } else {
            assert (0);
        }
    }

    /* Standard rasterization shader. Holds a reference to an image and a fixed foreground colour.
     * When a pixel is drawn, composit that pixel over the background image.*/
    struct CompositOverImageShader {
        io::Image<io::RGBA255>& image;
        const io::RGBA255& colour;

        CompositOverImageShader(io::Image<io::RGBA255>& image, const io::RGBA255& colour) : image(image), colour(colour) {}

        void render_pixel(int x, int y) {
            image.data[x + y * image.width] = composit_over_straight_255(image.data[x + y * image.width], colour);
        }
    };

    /* 2D Triangle Rasterization onto a given image. See `TriangleRasterizationMode` to select an implementation if desired. */
    template<TriangleRasterizationMode mode = TriangleRasterizationMode::Default>
    void rasterize_triangle_onto_image(const geometry2d::triangle& triangle, const io::RGBA255& colour, io::Image<io::RGBA255>& image) {
        CompositOverImageShader shader(image, colour);
        rasterize_triangle<CompositOverImageShader, mode>(triangle, image.width, image.height, shader);
    }

    /* 2D Polygon Rasterization onto a given image. See `rasterize_polygon_scanline` */
    inline void rasterize_polygon_onto_image(const std::vector<geometry2d::point>& polygon, const io::RGBA255& colour, io::Image<io::RGBA255>& image) {
        CompositOverImageShader shader(image, colour);
        rasterize_polygon_scanline(polygon, image.width, image.height, shader);
    }

    struct OptimalColourShader {
        long long int ab_r = 0;
        long long int ab_g = 0;
        long long int ab_b = 0;
        long long int a2 = 0;
        long long int b2 = 0;

        long long int error_delta = 0;

        unsigned char current_alpha;

        const io::Image<io::RGBA255>& target;
        const io::Image<io::RGBA255>& foreground;
        const io::Image<io::RGBA255>& background;
        const io::Image<int> error_mask;

        OptimalColourShader(unsigned char alpha, const io::Image<io::RGBA255>& target, const io::Image<io::RGBA255>& foreground, const io::Image<io::RGBA255>& background, const io::Image<int>& error_mask) 
            : current_alpha(alpha), target(target), foreground(foreground), background(background), error_mask(error_mask) {}

        void render_pixel(int x, int y) {
            io::RGBA255 target_pixel = target.data[y * target.width + x];
            io::RGBA255 background_pixel = background.data[y * target.width + x];
            io::RGBA255 foreground_pixel = foreground.data[y * target.width + x];

#define INT_MULT(a,b,t) ( (t) = (a) * (b) + 0x80, ( ( ( (t)>>8 ) + (t) )>>8 ) )
#define INT_LERP(p, q, a, t) ( (p) + INT_MULT( a, ( (q) - (p) ), t ) )

            long long int a_0, b_0_r, b_0_g, b_0_b;
            int t1, t2, t3;

            if (foreground_pixel.a == 0) {
                a_0 = current_alpha;
                b_0_r = int(target_pixel.r) - int(background_pixel.r) + INT_MULT(background_pixel.r, current_alpha, t1);
                b_0_g = int(target_pixel.g) - int(background_pixel.g) + INT_MULT(background_pixel.g, current_alpha, t1);
                b_0_b = int(target_pixel.b) - int(background_pixel.b) + INT_MULT(background_pixel.b, current_alpha, t1);
            } else {
                a_0 = int(current_alpha) - INT_MULT(current_alpha, foreground_pixel.a, t1);

                int tmp_r = int(background_pixel.r) - INT_MULT(background_pixel.r, current_alpha, t1);
                int tmp_g = int(background_pixel.g) - INT_MULT(background_pixel.g, current_alpha, t1);
                int tmp_b = int(background_pixel.b) - INT_MULT(background_pixel.b, current_alpha, t1);

                b_0_r = int(target_pixel.r) - INT_LERP(tmp_r, foreground_pixel.r, foreground_pixel.a, t2);
                b_0_g = int(target_pixel.g) - INT_LERP(tmp_g, foreground_pixel.g, foreground_pixel.a, t2);
                b_0_b = int(target_pixel.b) - INT_LERP(tmp_b, foreground_pixel.b, foreground_pixel.a, t2);
            }

            ab_r += a_0 * b_0_r;
            ab_g += a_0 * b_0_g;
            ab_b += a_0 * b_0_b;
            a2 += a_0 * a_0;
            b2 += b_0_r * b_0_r + b_0_g * b_0_g + b_0_b * b_0_b;

            error_delta -= error_mask.data[y * target.width + x];
        }

        std::pair<io::RGBA255, long long int> final_colour_and_error() const {
            if (a2 == 0) {
                return {io::RGBA255{0, 0, 0, 0}, 0};
            }

            unsigned char r = std::min(255, std::max(0, int((255 * ab_r) / a2)));
            unsigned char g = std::min(255, std::max(0, int((255 * ab_g) / a2)));
            unsigned char b = std::min(255, std::max(0, int((255 * ab_b) / a2)));
            auto out_colour = io::RGBA255{r, g, b, current_alpha};

            long long int error = error_delta + b2;
            error += ((int(r) * int(r) + int(g) * int(g) + int(b) * int(b)) * a2) / (255 * 255);
            error -= (2 * (int(r) * ab_r + int(g) * ab_g + int(b) * ab_b)) / 255;

            return {out_colour, error};
        }
    };

    struct DrawLossFGShader {
        const io::Image<io::RGBA255>& target;
        const io::Image<io::RGBA255>& foreground;
        const io::Image<io::RGBA255>& background;
        const io::Image<int>& error_mask;

        const io::RGBA255& colour;

        long long int total_error = 0;

        DrawLossFGShader(const io::Image<io::RGBA255>& target, const io::Image<io::RGBA255>& foreground, const io::Image<io::RGBA255>& background, const io::RGBA255& colour, const io::Image<int>& error_mask) 
            : target(target), foreground(foreground), background(background), colour(colour), error_mask(error_mask) {}

        void render_pixel(int x, int y) {
            io::RGBA255 target_pixel = target.data[y * target.width + x];
            io::RGBA255 background_pixel = background.data[y * target.width + x];
            io::RGBA255 foreground_pixel = foreground.data[y * target.width + x];

            auto middle_pixel = composit_over_straight_255(background_pixel, colour);
            auto final_pixel = composit_over_straight_255(middle_pixel, foreground_pixel);

            total_error += l2_difference(target_pixel, final_pixel);
            total_error -= error_mask.data[target.width * y + x];
        }
    };
};
