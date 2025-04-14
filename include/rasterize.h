#pragma once

#include "shaders.h"
#include "geometry.h"

#include "utils.h"

#include <iostream>
#include <memory>
#include <cstring>
#include <string>

enum class RasterStrategy {
    Bounded,
    Integer,
    Scanline,
    EdgeTable,
    Binned,
    Default = Bounded
};

std::string raster_strategy_name(RasterStrategy strategy) {
    switch (strategy) {
        case RasterStrategy::Bounded:
            return "Bounded";
        case RasterStrategy::Integer:
            return "Integer";
        case RasterStrategy::Scanline:
            return "Scanline";
        case RasterStrategy::EdgeTable:
            return "EdgeTable";
        case RasterStrategy::Binned:
            return "Binned";
        default:
            return "Unknown";
    }
}

struct RasterConfig {
    RasterStrategy strategy = RasterStrategy::Default;
    int image_width = 0;
    int image_height = 0;
};

template<class Shader>
void rasterize(std::shared_ptr<Shape> shape, Shader& shader, const RasterConfig& config) {
    ShapeType T = shape->type();

    if (config.strategy == RasterStrategy::Bounded) {
        if (T == ShapeType::Triangle) {
            rasterize_shape_bounded<Shader, Triangle>(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Circle) {
            rasterize_shape_bounded<Shader, Circle>(dynamic_cast<Circle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Polygon) {
            rasterize_shape_bounded<Shader, Polygon>(dynamic_cast<Polygon&>(*shape), config.image_width, config.image_height, shader);
        } else {
            throw std::runtime_error("Unsupported shape type for rasterization.");
        }
    } else if (config.strategy == RasterStrategy::Integer) {
        if (T != ShapeType::Triangle) {
            throw std::runtime_error("Integer rasterization strategy is only supported for triangles.");
        }
        rasterize_triangle_integer(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
    } else if (config.strategy == RasterStrategy::Scanline) {
        if (T == ShapeType::Triangle) {
            rasterize_polygon_scanline(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Polygon) {
            rasterize_polygon_scanline(dynamic_cast<Polygon&>(*shape), config.image_width, config.image_height, shader);
        } else {
            throw std::runtime_error("Scanline polygon rasterization is only supported for triangles and polygons.");
        }
    } else if (config.strategy == RasterStrategy::EdgeTable) {
        if (T == ShapeType::Triangle) {
            if constexpr (std::is_same_v<Shader, CompositOverShader>) {
                rasterize_test_polygon(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader); 
            } else {
                throw std::runtime_error("Test polygon rasterization is only supported for CompositOverShader.");
            }
        } else {
            throw std::runtime_error("Test polygon rasterization is only supported for triangles.");
        }
    } else if (config.strategy == RasterStrategy::Binned) {
        if (T == ShapeType::Triangle) {
            if constexpr (std::is_same_v<Shader, CompositOverShader>) {
                rasterize_triangle_binned(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
            } else {
                throw std::runtime_error("Test polygon rasterization is only supported for CompositOverShader.");
            }
        } else {
            throw std::runtime_error("Binned rasterization strategy is only supported for triangles.");
        }
    } else {
        throw std::runtime_error("Unsupported rasterization strategy.");
    }
}

/* Generic 2D Rasterization 
    * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `shape`
    * assuming an image domain of size `width` x `height`.
    * 
    * Method: Bounded loop.
    * - Determine the integer bounding box of the shape within the image
    * - For each pixel in the box, compute the barycentric coordinates at the center of that pixel w.r.t. the `shape`
    * - If the barycentric coordinates are all non-negative, shade the point. */
template<class Shader, typename Shape>
void rasterize_shape_bounded(const Shape& shape, int width, int height, Shader& shader) {
    Point lower = shape.min();
    Point upper = shape.max();

    int lower_x = std::max(0, std::min(int(lower.x * width), width - 1));
    int lower_y = std::max(0, std::min(int(lower.y * height), height - 1));
    int upper_x = std::max(0, std::min(int(upper.x * width), width - 1));
    int upper_y = std::max(0, std::min(int(upper.y * height), height - 1));

    for (int x = lower_x; x <= upper_x; x++) {
        for (int y = lower_y; y <= upper_y; y++) {
            float u = (x + 0.5f) / (float)width;
            float v = (y + 0.5f) / (float)height;
            if (shape.is_inside(Point{u, v})) {
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
void rasterize_triangle_integer(const Triangle& tri, int width, int height, Shader& shader) {
    int xs[3] = {int(tri[0].x * width), int(tri[1].x * width), int(tri[2].x * width)};
    int ys[3] = {int(tri[0].y * height), int(tri[1].y * height), int(tri[2].y * height)};

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
template<class Shader, typename Shape>
void rasterize_polygon_scanline(const Shape& polygon, int width, int height, Shader& shader) {
    constexpr int MAX_POLY_CORNERS = 10;

    int  nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap;

    if constexpr (!std::is_same_v<Shape, Triangle> && !std::is_same_v<Shape, Polygon>) {
        static_assert(false, "Unsupported shape type for scanline polygon rasterization.");
    }

    int polyCorners = polygon.num_vertices();
    if (polyCorners > MAX_POLY_CORNERS) {
        throw std::runtime_error("Polygon has too many corners.");
    }

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
                int pixelX = nodeX[i];
                if constexpr (std::is_same_v<Shader, CompositOverShader>) {
                    int base_ptr_delta = (long int)(shader.background.data()) % 4;
                    while ((base_ptr_delta + pixelX) % 4 != 0 && pixelX <= nodeX[i+1]) {
                        shader.render_pixel(pixelX, pixelY);
                        pixelX++;
                    }
                    while (pixelX + 4 <= nodeX[i+1]) {
                        // sse 4x shading
                        unsigned char* ptr = (unsigned char*)(shader.background.data() + pixelY * width + pixelX);
                        __m128i newVal = alphaBlendSSE(
                            ptr,
                            shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
                        _mm_store_si128((__m128i*)ptr, newVal);
                        pixelX += 4;
                    }
                }

                while (pixelX <= nodeX[i+1]) {
                    shader.render_pixel(pixelX, pixelY);
                    pixelX++;
                }
            }
        }
    }
}

template<class Shader>
int ImagingDrawPolygon(int count, int *xy, int width, int height, Shader& shader);

// template<class Shader>
void rasterize_test_polygon(const Triangle& tri, int width, int height, CompositOverShader& shader) {
    int pts[6] = {
        int(tri[0].x * width), int(tri[0].y * height),
        int(tri[1].x * width), int(tri[1].y * height),
        int(tri[2].x * width), int(tri[2].y * height)
    };
    ImagingDrawPolygon(3, &pts[0], width, height, shader);
}

/* TEST CODE IMPORTED FROM libImaging (from pillow's PIL.ImageDraw core) */


#include <math.h>
#include <stdint.h>

#define CEIL(v) (int)ceil(v)
#define FLOOR(v) ((v) >= 0.0 ? (int)(v) : (int)floor(v))

#define SHIFTFORDIV255(a) ((((a) >> 8) + a) >> 8)

/* like (a * b + 127) / 255), but much faster on most platforms */
#define MULDIV255(a, b, tmp) (tmp = (a) * (b) + 128, SHIFTFORDIV255(tmp))

#define DIV255(a, tmp) (tmp = (a) + 128, SHIFTFORDIV255(tmp))

#define BLEND(mask, in1, in2, tmp1) DIV255(in1 * (255 - mask) + in2 * mask, tmp1)

#define PREBLEND(mask, in1, in2, tmp1) (MULDIV255(in1, (255 - mask), tmp1) + in2)

#define CLIP8(v) ((v) <= 0 ? 0 : (v) < 256 ? (v) : 255)

/*
 * Rounds around zero (up=away from zero, down=towards zero)
 * This guarantees that ROUND_UP|DOWN(f) == -ROUND_UP|DOWN(-f)
 */
#define ROUND_UP(f) ((int)((f) >= 0.0 ? floor((f) + 0.5F) : -floor(fabs(f) + 0.5F)))
#define ROUND_DOWN(f) ((int)((f) >= 0.0 ? ceil((f) - 0.5F) : -ceil(fabs(f) - 0.5F)))

typedef struct {
    /* edge descriptor for polygon engine */
    int d;
    int x0, y0;
    int xmin, ymin, xmax, ymax;
    float dx;
} Edge;


template<class Shader>
static inline void
point32rgba(int x, int y, int width, int height, Shader& shader) {
    unsigned int tmp;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        shader.render_pixel(x, y);
    }
}

template<class Shader>
static inline void
hline32rgba(int x0, int y0, int x1, int width, int height, Shader& shader) {
    unsigned int tmp;

    if (y0 >= 0 && y0 < height) {
        if (x0 < 0) {
            x0 = 0;
        } else if (x0 >= width) {
            return;
        }
        if (x1 < 0) {
            return;
        } else if (x1 >= width) {
            x1 = width - 1;
        }
        if (x0 <= x1) {
            int base_ptr_delta = (long int)(shader.background.data()) % 4;
            while ((base_ptr_delta + x0) % 4 != 0 && x0 <= x1) {
                shader.render_pixel(x0, y0);
                x0++;
            }
            // while (x0 + 8 <= x1) {
            //     // avx2 8x shading
            //     unsigned char* ptr = (unsigned char*)(shader.background.data() + y0 * width + x0);
            //     __m256i newVal = alphaBlendAVX2_SinglePath(
            //         ptr,
            //         shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
            //     _mm256_store_si256((__m256i*)ptr, newVal);
            //     x0 += 8;
            // }
            while (x0 + 4 <= x1) {
                // sse 4x shading
                unsigned char* ptr = (unsigned char*)(shader.background.data() + y0 * width + x0);
                __m128i newVal = alphaBlendSSE(
                    ptr,
                    shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
                _mm_store_si128((__m128i*)ptr, newVal);
                x0 += 4;
            }

            while (x0 <= x1) {
                shader.render_pixel(x0, y0);
                x0++;
            }
        }
    }
}

template<class Shader>
static inline void
line32rgba(int x0, int y0, int x1, int y1, int width, int height, Shader& shader) {
    int i, n, e;
    int dx, dy;
    int xs, ys;

    /* normalize coordinates */
    dx = x1 - x0;
    if (dx < 0) {
        dx = -dx, xs = -1;
    } else {
        xs = 1;
    }
    dy = y1 - y0;
    if (dy < 0) {
        dy = -dy, ys = -1;
    } else {
        ys = 1;
    }

    n = (dx > dy) ? dx : dy;

    if (dx == 0) {
        /* vertical */
        for (i = 0; i < dy; i++) {
            point32rgba(x0, y0, width, height, shader);
            y0 += ys;
        }
    } else if (dy == 0) {
        /* horizontal */
        for (i = 0; i < dx; i++) {
            point32rgba(x0, y0, width, height, shader);
            x0 += xs;
        }
    } else if (dx > dy) {
        /* bresenham, horizontal slope */
        n = dx;
        dy += dy;
        e = dy - dx;
        dx += dx;

        for (i = 0; i < n; i++) {
            point32rgba(x0, y0, width, height, shader);
            if (e >= 0) {
                y0 += ys;
                e -= dx;
            }
            e += dy;
            x0 += xs;
        }
    } else {
        /* bresenham, vertical slope */
        n = dy;
        dx += dx;
        e = dx - dy;
        dy += dy;

        for (i = 0; i < n; i++) {
            point32rgba(x0, y0, width, height, shader);
            if (e >= 0) {
                x0 += xs;
                e -= dy;
            }
            e += dx;
            y0 += ys;
        }
    }
}

static int
x_cmp(const void *x0, const void *x1) {
    float diff = *((float *)x0) - *((float *)x1);
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}

template<class Shader>
static void
draw_horizontal_lines(
    int n, Edge *e, int *x_pos, int y, int width, int height, Shader& shader
) {
    int i;
    for (i = 0; i < n; i++) {
        if (e[i].ymin == y && e[i].ymin == e[i].ymax) {
            int xmax;
            int xmin = e[i].xmin;
            if (*x_pos != -1 && *x_pos < xmin) {
                // Line would be after the current position
                continue;
            }

            xmax = e[i].xmax;
            if (*x_pos > xmin) {
                // Line would be partway through x_pos, so increase the starting point
                xmin = *x_pos;
                if (xmax < xmin) {
                    // Line would now end before it started
                    continue;
                }
            }

            hline32rgba(xmin, e[i].ymin, xmax, width, height, shader);
            *x_pos = xmax + 1;
        }
    }
}

/*
 * Filled polygon draw function using scan line algorithm.
 */
template<class Shader>
static inline int polygon32rgba(int n, Edge *e, int width, int height, Shader& shader) {
    Edge **edge_table;
    float *xx;
    int edge_count = 0;
    int ymin = height - 1;
    int ymax = 0;
    int i, j, k;
    float adjacent_line_x, adjacent_line_x_other_edge;

    if (n <= 0) {
        return 0;
    }

    /* Initialize the edge table and find polygon boundaries */
    /* malloc check ok, using calloc */
    edge_table = (Edge**) calloc(n, sizeof(Edge *));
    if (!edge_table) {
        return -1;
    }

    for (i = 0; i < n; i++) {
        if (ymin > e[i].ymin) {
            ymin = e[i].ymin;
        }
        if (ymax < e[i].ymax) {
            ymax = e[i].ymax;
        }
        if (e[i].ymin == e[i].ymax) {
            continue;
        }
        edge_table[edge_count++] = (e + i);
    }
    if (ymin < 0) {
        ymin = 0;
    }
    if (ymax > height) {
        ymax = height;
    }

    /* Process the edge table with a scan line searching for intersections */
    /* malloc check ok, using calloc */
    xx = (float*) calloc(edge_count * 2, sizeof(float));
    if (!xx) {
        free(edge_table);
        return -1;
    }
    for (; ymin <= ymax; ymin++) {
        j = 0;
        for (i = 0; i < edge_count; i++) {
            Edge *current = edge_table[i];
            if (ymin >= current->ymin && ymin <= current->ymax) {
                xx[j++] = (ymin - current->y0) * current->dx + current->x0;

                if (ymin == current->ymax && ymin < ymax) {
                    // Needed to draw consistent polygons
                    xx[j] = xx[j - 1];
                    j++;
                } else if (current->dx != 0 && j % 2 == 1 &&
                           roundf(xx[j - 1]) == xx[j - 1]) {
                    // Connect discontiguous corners
                    for (k = 0; k < i; k++) {
                        Edge *other_edge = edge_table[k];
                        if ((current->dx > 0 && other_edge->dx <= 0) ||
                            (current->dx < 0 && other_edge->dx >= 0)) {
                            continue;
                        }
                        // Check if the two edges join to make a corner
                        if (xx[j - 1] ==
                            (ymin - other_edge->y0) * other_edge->dx + other_edge->x0) {
                            // Determine points from the edges on the next row
                            // Or if this is the last row, check the previous row
                            int offset = ymin == ymax ? -1 : 1;
                            adjacent_line_x =
                                (ymin + offset - current->y0) * current->dx +
                                current->x0;
                            adjacent_line_x_other_edge =
                                (ymin + offset - other_edge->y0) * other_edge->dx +
                                other_edge->x0;
                            if (ymin == current->ymax) {
                                if (current->dx > 0) {
                                    xx[k] =
                                        fmax(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) +
                                        1;
                                } else {
                                    xx[k] =
                                        fmin(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) -
                                        1;
                                }
                            } else {
                                if (current->dx > 0) {
                                    xx[k] = fmin(
                                        adjacent_line_x, adjacent_line_x_other_edge
                                    );
                                } else {
                                    xx[k] =
                                        fmax(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) +
                                        1;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        qsort(xx, j, sizeof(float), x_cmp);
        int x_pos = j == 0 ? -1 : 0;
        for (i = 1; i < j; i += 2) {
            int x_end = ROUND_DOWN(xx[i]);
            if (x_end < x_pos) {
                // Line would be before the current position
                continue;
            }
            draw_horizontal_lines(n, e, &x_pos, ymin, width, height, shader);
            if (x_end < x_pos) {
                // Line would be before the current position
                continue;
            }

            int x_start = ROUND_UP(xx[i - 1]);
            if (x_pos > x_start) {
                // Line would be partway through x_pos, so increase the starting
                // point
                x_start = x_pos;
                if (x_end < x_start) {
                    // Line would now end before it started
                    continue;
                }
            }
            hline32rgba(x_start, ymin, x_end, width, height, shader);
            x_pos = x_end + 1;
        }
        draw_horizontal_lines(n, e, &x_pos, ymin, width, height, shader);
    }

    free(xx);
    free(edge_table);
    return 0;
}

static inline void
add_edge(Edge *e, int x0, int y0, int x1, int y1) {
    /* printf("edge %d %d %d %d\n", x0, y0, x1, y1); */

    if (x0 <= x1) {
        e->xmin = x0, e->xmax = x1;
    } else {
        e->xmin = x1, e->xmax = x0;
    }

    if (y0 <= y1) {
        e->ymin = y0, e->ymax = y1;
    } else {
        e->ymin = y1, e->ymax = y0;
    }

    if (y0 == y1) {
        e->d = 0;
        e->dx = 0.0;
    } else {
        e->dx = ((float)(x1 - x0)) / (y1 - y0);
        if (y0 == e->ymin) {
            e->d = 1;
        } else {
            e->d = -1;
        }
    }

    e->x0 = x0;
    e->y0 = y0;
}

/* -------------------------------------------------------------------- */
/* Interface                                                            */
/* -------------------------------------------------------------------- */

template<class Shader>
int ImagingDrawPolygon(int count, int *xy, int width, int height, Shader& shader) {
    int i, n, x0, y0, x1, y1;

    if (count <= 0) {
        return 0;
    }

    /* Build edge list */
    /* malloc check ok, using calloc */
    Edge *e = (Edge*) calloc(count, sizeof(Edge));
    if (!e) {
        return -1;
    }
    for (i = n = 0; i < count - 1; i++) {
        x0 = xy[i * 2];
        y0 = xy[i * 2 + 1];
        x1 = xy[i * 2 + 2];
        y1 = xy[i * 2 + 3];
        if (y0 == y1 && i != 0 && y0 == xy[i * 2 - 1]) {
            // This is a horizontal line,
            // that immediately follows another horizontal line
            Edge *last_e = &e[n - 1];
            if (x1 > x0 && x0 > xy[i * 2 - 2]) {
                // They are both increasing in x
                last_e->xmax = x1;
                continue;
            } else if (x1 < x0 && x0 < xy[i * 2 - 2]) {
                // They are both decreasing in x
                last_e->xmin = x1;
                continue;
            }
        }
        add_edge(&e[n++], x0, y0, x1, y1);
    }
    if (xy[i * 2] != xy[0] || xy[i * 2 + 1] != xy[1]) {
        add_edge(&e[n++], xy[i * 2], xy[i * 2 + 1], xy[0], xy[1]);
    }
    polygon32rgba(n, e, width, height, shader);
    free(e);

    return 0;
}

void rasterize_triangle_binned(const Triangle& tri, int width, int height, CompositOverShader& shader) {
    // Convert triangle vertices to screen space coordinates.
    float pts[6] = {
        tri[0].x * width, tri[0].y * height,
        tri[1].x * width, tri[1].y * height,
        tri[2].x * width, tri[2].y * height
    };

    // Construct vec2's for use with the edge functions.
    vec2 v0(pts[0], pts[1]);
    vec2 v1(pts[2], pts[3]);
    vec2 v2(pts[4], pts[5]);

    // Compute bounding box using integer approximations.
    int pts_int[6] = {
        int(pts[0]), int(pts[1]),
        int(pts[2]), int(pts[3]),
        int(pts[4]), int(pts[5])
    };

    int xmin = std::min({ pts_int[0], pts_int[2], pts_int[4] });
    int xmax = std::max({ pts_int[0], pts_int[2], pts_int[4] });
    int ymin = std::min({ pts_int[1], pts_int[3], pts_int[5] });
    int ymax = std::max({ pts_int[1], pts_int[3], pts_int[5] });

    xmin = clampi(xmin, 0, width - 1);
    xmax = clampi(xmax + 1, 0, width - 1);
    ymin = clampi(ymin, 0, height - 1);
    ymax = clampi(ymax + 1, 0, height - 1);

    constexpr int tile_size = 4;

    xmin = xmin - (xmin % 4);

    unsigned char* framebuffer = (unsigned char*) shader.background.data();

    float filledbitsfloat;
	{
		unsigned int ii = 0xffffffff;
		memcpy(&filledbitsfloat, &ii, sizeof(float));
	}
	float whitecolorfloat = filledbitsfloat;

    __m128 zero = _mm_setzero_ps();

    // Loop over tiles in the bounding box.
    for (int y_tile_start = ymin; y_tile_start < ymax; y_tile_start += tile_size) {
        for (int x_tile_start = xmin; x_tile_start < xmax; x_tile_start += tile_size) {
            // Compute tile corner coordinates (in screen space).
            float x0 = float(x_tile_start);
            float y0 = float(y_tile_start);
            float x1 = float(x_tile_start + tile_size);
            float y1 = float(y_tile_start + tile_size);

            // Pack the 4 corner coordinates into SSE registers.
            // Order: top-left, top-right, bottom-right, bottom-left.
            __m128 cx = _mm_setr_ps(x0, x1, x1, x0);
            __m128 cy = _mm_setr_ps(y0, y0, y1, y1);

            // --- Edge 0: from v0 to v1 ---
            __m128 edge0_vals = edgeFunctionSSE(v0, v1, cx, cy);
            // Compute horizontal minimum using one movehl and one shuffle.
            __m128 t0 = _mm_min_ps(edge0_vals, _mm_movehl_ps(edge0_vals, edge0_vals));
            __m128 t0_shuf = _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(0,0,0,1));
            float min_e0 = _mm_cvtss_f32(_mm_min_ss(t0, t0_shuf));
            // And horizontal maximum.
            __m128 t0_max = _mm_max_ps(edge0_vals, _mm_movehl_ps(edge0_vals, edge0_vals));
            __m128 t0_max_shuf = _mm_shuffle_ps(t0_max, t0_max, _MM_SHUFFLE(0,0,0,1));
            float max_e0 = _mm_cvtss_f32(_mm_max_ss(t0_max, t0_max_shuf));

            // --- Edge 1: from v1 to v2 ---
            __m128 edge1_vals = edgeFunctionSSE(v1, v2, cx, cy);
            __m128 t1 = _mm_min_ps(edge1_vals, _mm_movehl_ps(edge1_vals, edge1_vals));
            __m128 t1_shuf = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(0,0,0,1));
            float min_e1 = _mm_cvtss_f32(_mm_min_ss(t1, t1_shuf));
            __m128 t1_max = _mm_max_ps(edge1_vals, _mm_movehl_ps(edge1_vals, edge1_vals));
            __m128 t1_max_shuf = _mm_shuffle_ps(t1_max, t1_max, _MM_SHUFFLE(0,0,0,1));
            float max_e1 = _mm_cvtss_f32(_mm_max_ss(t1_max, t1_max_shuf));

            // --- Edge 2: from v2 to v0 ---
            __m128 edge2_vals = edgeFunctionSSE(v2, v0, cx, cy);
            __m128 t2 = _mm_min_ps(edge2_vals, _mm_movehl_ps(edge2_vals, edge2_vals));
            __m128 t2_shuf = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0,0,0,1));
            float min_e2 = _mm_cvtss_f32(_mm_min_ss(t2, t2_shuf));
            __m128 t2_max = _mm_max_ps(edge2_vals, _mm_movehl_ps(edge2_vals, edge2_vals));
            __m128 t2_max_shuf = _mm_shuffle_ps(t2_max, t2_max, _MM_SHUFFLE(0,0,0,1));
            float max_e2 = _mm_cvtss_f32(_mm_max_ss(t2_max, t2_max_shuf));

            // Conservative tile classification:
            // If any edge's maximum value is below 0, the tile is completely outside.
            if ((max_e0 < 0) || (max_e1 < 0) || (max_e2 < 0)) {
                continue;
            }
            // If all edges have a minimum value >= 0, the tile is completely inside.
            bool tileFullyInside = (min_e0 >= 0 && min_e1 >= 0 && min_e2 >= 0);

            if (tileFullyInside) {
                // Fast fill: shade every pixel in the tile.
                for (int y_idx = 0; y_idx < tile_size; y_idx++) {
                    for (int x_idx = 0; x_idx < tile_size; x_idx += 4) {
                        unsigned char* ptr = (unsigned char*)(framebuffer + 4 * ((y_tile_start + y_idx) * width + x_tile_start + x_idx));

                        __m128i newBufferVal = alphaBlendSSE(ptr, shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
                        _mm_store_si128((__m128i*)ptr, newBufferVal);	
                    }
                }
            } else {
                // Partially covered tile: test each pixel individually.
                for (int y_idx = 0; y_idx < tile_size; y_idx++) {
                    float v = (y_tile_start + y_idx + 0.5f);
                    __m128 py = _mm_set1_ps(v);
                    for (int x_idx = 0; x_idx < tile_size; x_idx += 4) {
                        unsigned char* ptr = (unsigned char*)(framebuffer + 4 * ((y_tile_start + y_idx) * width + x_tile_start + x_idx));
                        float u = x_tile_start + x_idx + 0.5f;
                        __m128 px = _mm_set_ps(u + 3.0f, u + 2.0f, u + 1.0f, u + 0.0f);
                        // px = _mm_mul_ps(px, _mm_set1_ps(1.0f / width));

                        __m128 w0 = edgeFunctionSSE(v1, v2, px, py);
                        __m128 w1 = edgeFunctionSSE(v2, v0, px, py);
                        __m128 w2 = edgeFunctionSSE(v0, v1, px, py);

                        // the default bitflag, results in all the four pixels being overwritten.
                        __m128 writeFlag = _mm_set_ps1(filledbitsfloat);

                        // the results of the edge tests are used to modify our bitflag.
                        writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w0, zero));
                        writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w1, zero));
                        writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w2, zero));

                        __m128i origBufferVal = _mm_load_si128((__m128i*)ptr);
                        __m128i newBufferVal = alphaBlendSSE(ptr, shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
                        _mm_store_si128((__m128i*)ptr,
                            _mm_or_si128(
                                _mm_and_si128(_mm_castps_si128(writeFlag), newBufferVal),
                                _mm_andnot_si128(_mm_castps_si128(writeFlag), origBufferVal)
                            ));	
                    }
                }
            }
        }
    }
}

void rasterize_triangle_avx2_integer(const Triangle& tri, int width, int height, CompositOverShader& shader) {
    // Convert triangle vertices to integer screen coordinates.
    int xs[3] = { int(tri[0].x * width), int(tri[1].x * width), int(tri[2].x * width) };
    int ys[3] = { int(tri[0].y * height), int(tri[1].y * height), int(tri[2].y * height) };

    // Orient the triangle: if the area is negative, swap the first two vertices.
    int w = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]);
    if (w < 0) {
        std::swap(xs[0], xs[1]);
        std::swap(ys[0], ys[1]);
    }

    // Compute the bounding box of the triangle.
    int lower_x = std::max(0, std::min({ xs[0], xs[1], xs[2] }));
    int lower_y = std::max(0, std::min({ ys[0], ys[1], ys[2] }));
    int upper_x = std::min(width - 1, std::max({ xs[0], xs[1], xs[2] }));
    int upper_y = std::min(height - 1, std::max({ ys[0], ys[1], ys[2] }));

    unsigned char* framebuffer = (unsigned char*) shader.background.data();
    lower_x = lower_x - (lower_x % 8);

    // Precompute coefficients for the three edge functions.
    // We use the following definitions:
    //   w0(P) = orient2d(V1, V2, P)
    //   w1(P) = orient2d(V2, V0, P)
    //   w2(P) = orient2d(V0, V1, P)
    int A01 = ys[0] - ys[1], B01 = xs[1] - xs[0];
    int A12 = ys[1] - ys[2], B12 = xs[2] - xs[1];
    int A20 = ys[2] - ys[0], B20 = xs[0] - xs[2];

    auto orient2d = [](int ax, int ay, int bx, int by, int cx, int cy) -> int {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    };

    // Compute the starting edge values at the upper-left corner of the bounding box.
    int w0_row = orient2d(xs[1], ys[1], xs[2], ys[2], lower_x, lower_y);
    int w1_row = orient2d(xs[2], ys[2], xs[0], ys[0], lower_x, lower_y);
    int w2_row = orient2d(xs[0], ys[0], xs[1], ys[1], lower_x, lower_y);

    // For horizontal stepping, each pixel increases the edge functions by:
    //    w0: +A12,  w1: +A20,  w2: +A01.
    // For vertical stepping (per scanline), the starting values are incremented by:
    //    w0: +B12,  w1: +B20,  w2: +B01.

    // Prepare the constant source color (compositing colour) in packed 32-bit format.
    uint32_t src_color_int = (shader.colour.a << 24) | (shader.colour.r << 16) |
                               (shader.colour.g << 8)  | (shader.colour.b);
    __m256i src_color = _mm256_set1_epi32(src_color_int);

    // A vector of offsets [0, 1, 2, 3, 4, 5, 6, 7] for processing eight pixels.
    __m256i offset_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i step_w0 = _mm256_set1_epi32(A12);
    __m256i step_w1 = _mm256_set1_epi32(A20);
    __m256i step_w2 = _mm256_set1_epi32(A01);

    // Get pointer to the framebuffer.

    // Loop over each scanline in the bounding box.
    for (int y = lower_y; y <= upper_y; y++) {
        int w0_cur = w0_row;
        int w1_cur = w1_row;
        int w2_cur = w2_row;

        // Process in blocks of 8 pixels.
        for (int x = lower_x; x <= upper_x; x += 8) {
            // Build the w0..w2 vectors from these accumulators.
            __m256i w0_vec = _mm256_setr_epi32(
                w0_cur, w0_cur + A12, w0_cur + 2*A12, w0_cur + 3*A12,
                w0_cur + 4*A12, w0_cur + 5*A12, w0_cur + 6*A12, w0_cur + 7*A12
            );
            __m256i w1_vec = _mm256_setr_epi32(
                w1_cur, w1_cur + A20, w1_cur + 2*A20, w1_cur + 3*A20,
                w1_cur + 4*A20, w1_cur + 5*A20, w1_cur + 6*A20, w1_cur + 7*A20
            );
            __m256i w2_vec = _mm256_setr_epi32(
                w2_cur, w2_cur + A01, w2_cur + 2*A01, w2_cur + 3*A01,
                w2_cur + 4*A01, w2_cur + 5*A01, w2_cur + 6*A01, w2_cur + 7*A01
            );

            // Determine which pixels are inside.
            __m256i mask0 = _mm256_cmpgt_epi32(w0_vec, _mm256_set1_epi32(-1));
            __m256i mask1 = _mm256_cmpgt_epi32(w1_vec, _mm256_set1_epi32(-1));
            __m256i mask2 = _mm256_cmpgt_epi32(w2_vec, _mm256_set1_epi32(-1));
            __m256i in_mask = _mm256_and_si256(mask0, _mm256_and_si256(mask1, mask2));

            int pixel_index = y * width + x;
            unsigned char* ptr = framebuffer + pixel_index * 4;

            __m256i dest_pixels = _mm256_load_si256((__m256i const*)ptr);
            __m256i blended_pixels = alphaBlendAVX2_SinglePath(
                ptr, shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a
            );
            __m256i result_pixels = _mm256_blendv_epi8(dest_pixels, blended_pixels, in_mask);
            _mm256_store_si256((__m256i*)ptr, result_pixels);

            // Advance accumulators by 8 steps horizontally.
            w0_cur += 8*A12;
            w1_cur += 8*A20;
            w2_cur += 8*A01;
        }

        // Advance the row starting values by the vertical increments.
        w0_row += B12;
        w1_row += B20;
        w2_row += B01;
    }
}