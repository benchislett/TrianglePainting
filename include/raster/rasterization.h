#pragma once

#include "geometry/types.h"
#include "io/image.h"
#include "raster/composit.h"

#include <vector>

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
                float u = (x + 0.5f) / (float)image.width;
                float v = (y + 0.5f) / (float)image.height;
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
    void rasterize_polygon(const std::vector<geometry2d::point>& polygon, int width, int height, Shader& shader) {
        constexpr int MAX_POLY_CORNERS = 10;

        int  nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap ;

        int polyCorners = polygon.size();

        //  Loop through the rows of the image.
        for (pixelY=0; pixelY<height; pixelY++) {

            //  Build a list of nodes.
            nodes=0; j=polyCorners-1;
            for (i=0; i<polyCorners; i++) {
                if (polygon[i].y<(double) pixelY && polygon[j].y>=(double) pixelY
                ||  polygon[j].y<(double) pixelY && polygon[i].y>=(double) pixelY) {
                nodeX[nodes++]=(int) (polygon[i].x+(pixelY-polygon[i].y)/(polygon[j].y-polygon[i].y)
                *(polygon[j].x-polygon[i].x)); }
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

    inline void rasterize_triangles_using_generic_shader(const RasterScene& scene, io::Image<io::RGBA255>& image) {
        struct Shader {
            const RasterScene& scene;
            io::Image<io::RGBA255>& image;
            int triangle_index;

            Shader(const RasterScene& scene, io::Image<io::RGBA255>& image, int triangle_index) : scene(scene), image(image), triangle_index(triangle_index) {}

            void render_pixel(int x, int y) {
                image.data[x + y * image.width] = composit_over_straight_255(image.data[x + y * image.width], scene.colours[triangle_index]);
            }
        };

        std::fill(image.data.begin(), image.data.end(), scene.background_colour);
        for (int i = 0; i < scene.triangles.size(); i++) {
            Shader tmp(scene, image, i);
            // rasterize_triangle<Shader>(scene.triangles[i], image.width, image.height, tmp);

            std::vector<geometry2d::point> points_rescaled = {geometry2d::point{scene.triangles[i].a.x * image.width, scene.triangles[i].a.y * image.height},
                                                              geometry2d::point{scene.triangles[i].b.x * image.width, scene.triangles[i].b.y * image.height},
                                                              geometry2d::point{scene.triangles[i].c.x * image.width, scene.triangles[i].c.y * image.height}};
            rasterize_polygon(points_rescaled, image.width, image.height, tmp);
        }
    }
};
