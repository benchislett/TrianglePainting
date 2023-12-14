module Raster2D

using Images

using ..Shapes2D
using ..Spatial2D
using ..Pixel

export u2x, v2y, x2u, y2v
export RasterState, rasterfunc, rasterize
export RasterAlgorithm, RasterAlgorithmScanline, RasterAlgorithmBounded, RasterAlgorithmPointwise

"""Convert a UV-space U coordinate to an image-space X coordinate"""
u2x(u, w) = trunc(Int32, w * u)

"""Convert a UV-space V coordinate to an image-space Y coordinate"""
v2y(v, h) = trunc(Int32, h * v)

"""Convert an image-space X coordinate to a UV-space U coordinate"""
x2u(x, w) = Float32(x) / Float32(w)

"""Convert an image-space Y coordinate to a UV-space V coordinate"""
y2v(y, h) = Float32(y) / Float32(h)

"""Stateful struct to be rasterized over the pixels of a shape by `rasterfunc`"""
abstract type RasterState end

"""
    rasterfunc(i, j, image, state)

Advance the rasterization state of type RasterState at pixel (i, j) and return the new state.
"""
rasterfunc(i, j, image, ::RasterState)::RasterState = error("Not implemented")

"""
Algorithm to use for rasterization.

One of [`RasterAlgorithmScanline`](@ref), [`RasterAlgorithmBounded`](@ref), [`RasterAlgorithmPointwise`](@ref).
"""
abstract type RasterAlgorithm end

"""Rasterize a triangle in scanlines, analytically intersecting with the edges"""
struct RasterAlgorithmScanline <: RasterAlgorithm end

"""Rasterize a bounded shape by iterating over its bounding box"""
struct RasterAlgorithmBounded <: RasterAlgorithm end

"""Rasterize a general shape by iterating over all pixels in the image"""
struct RasterAlgorithmPointwise <: RasterAlgorithm end

"""
    rasterize(shape, image, state, algorithm)

Rasterize the shape over the image.
For each pixel to be rasterized, advance the state. Return the final state.

See also [`RasterAlgorithm`](@ref)
As well as usage mechanisms [`RasterState`](@ref) and [`rasterfunc`](@ref)
"""
rasterize(shape::AbstractShape, image, state::RasterState, ::RasterAlgorithm) = error("Not implemented")

function rasterize(shape, image, state, ::RasterAlgorithmPointwise)
    w, h = size(image)

    for y = 1:h
        for x = 1:w
            if covers(shape, Point(x2u(x, w), y2v(y, h)))
                state = rasterfunc(x, y, image, state)
            end
        end
    end
    state
end

function rasterize(shape, image, state, ::RasterAlgorithmBounded)
    bbox = AABB(shape)
    w, h = size(image)

    for y in max(1, v2y(y(bbox.min), h)):min(h, v2y(y(bbox.max), h) + 1)
        for x in max(1, u2x(x(bbox.min), w)):min(w, u2x(x(bbox.max), w) + 1)
            if covers(shape, Point(x2u(x, w), y2v(y, h)))
                state = rasterfunc(x, y, image, state)
            end
        end
    end
    state
end

"""
Rasterize the top half of a triangle
Ported from:
http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
"""
function rasttop(shape::Triangle, image, state)
    w, h = size(image)

    v1x, v1y = vertex(shape, 1)
    v2x, v2y = vertex(shape, 2)
    v3x, v3y = vertex(shape, 3)

    invslope1::Float32 = (v3x - v1x) / (v3y - v1y)
    invslope2::Float32 = (v3x - v2x) / (v3y - v2y)

    if invslope1 <= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = v3x
    curx2::Float32 = v3x

    y = v2y(v3y, h)
    if y > h
        curx1 -= (invslope1 * (y - h)) / Float32(h)
        curx2 -= (invslope2 * (y - h)) / Float32(h)
        y = h
    end
    while y >= max(1, v2y(v1y, h))
        for x = max(1, u2x(curx1, w)):min(w, u2x(curx2, w))
            state = rasterfunc(x, y, image, state)
        end
        y = y - 1
        curx1 -= invslope1 / Float32(h)
        curx2 -= invslope2 / Float32(h)
    end

    state
end

"""
Rasterize the bottom half of a triangle
Ported from:
http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
"""
function rastbot(shape::Triangle, image, state)
    w, h = size(image)

    v1x, v1y = vertex(shape, 1)
    v2x, v2y = vertex(shape, 2)
    v3x, v3y = vertex(shape, 3)

    invslope1::Float32 = (v2x - v1x) / (v2y - v1y)
    invslope2::Float32 = (v3x - v1x) / (v3y - v1y)

    if invslope1 >= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = v1x
    curx2::Float32 = v1x

    y = v2y(v1y, h)
    if y < 1
        curx1 -= (invslope1 * (1 - y)) / Float32(h)
        curx2 -= (invslope2 * (1 - y)) / Float32(h)
        y = 1
    end
    while y <= min(h, v2y(v2y, h))
        for x = max(1, u2x(curx1, w)):min(w, u2x(curx2, w))
            state = rasterfunc(x, y, image, state)
        end
        y = y + 1
        curx1 += invslope1 / Float32(h)
        curx2 += invslope2 / Float32(h)
    end

    state
end

function rasterize(tri::Triangle, image, state, ::RasterAlgorithmScanline)
    v1, v2, v3 = vertices(tri)

    # sort vertices in ascending order
    if v1.second >= v2.second
        v1, v2 = v2, v1
    end

    if v1.second >= v3.second
        v1, v3 = v3, v1
    end

    if v2.second >= v3.second
        v2, v3 = v3, v2
    end

    # Save work if the triangle is already flat
    if v2.second == v3.second
        state = rastbot(Triangle(v1, v2, v3), image, state)
    elseif v1.second == v2.second
        state = rasttop(Triangle(v1, v2, v3), image, state)
    else
        x4 = v1.first + ((v2.second - v1.second) / (v3.second - v1.second)) * (v3.first - v1.first)
        v4 = Point(x4, v2.second)
        state = rastbot(Triangle(v1, v2, v4), image, state)
        state = rasttop(Triangle(v2, v4, v3), image, state)
    end
    state
end

end