module Raster2D

using Images

using ..Shapes2D
using ..Spatial2D
using ..Pixel

export u2x, v2y, x2u, y2v
export RasterState, rasterfunc, rasterize
export RasterAlgorithm, RasterAlgorithmScanline, RasterAlgorithmBounded, RasterAlgorithmPointwise

"""Convert a UV-space U coordinate to an image-space X coordinate"""
u2x(u, w) = trunc(Int32, w * u) + 1

"""Convert a UV-space V coordinate to an image-space Y coordinate"""
v2y(v, h) = trunc(Int32, h * v) + 1

"""Convert an image-space X coordinate to a UV-space U coordinate"""
x2u(x, w) = Float32(x - 1) / Float32(w)

"""Convert an image-space Y coordinate to a UV-space V coordinate"""
y2v(y, h) = Float32(y - 1) / Float32(h)

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
    rasterize(image, shape, state, algorithm)

Rasterize the shape over the image.
For each pixel to be rasterized, advance the state. Return the final state.

See also [`RasterAlgorithm`](@ref)
As well as usage mechanisms [`RasterState`](@ref) and [`rasterfunc`](@ref)
"""
rasterize(image, shape, state, ::RasterAlgorithm) = error("Not implemented")

function rasterize(image, shape, state, ::RasterAlgorithmPointwise)
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

function rasterize(image, shape, state, ::RasterAlgorithmBounded)
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
function rasttop(image, shape::Triangle, state)
    w, h = size(image)

    v1, v2, v3 = vertices(shape)

    invslope1::Float32 = (x(v3) - x(v1)) / (y(v3) - y(v1))
    invslope2::Float32 = (x(v3) - x(v2)) / (y(v3) - y(v2))

    if invslope1 <= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = x(v3)
    curx2::Float32 = x(v3)

    yval = v2y(y(v3), h)
    if yval > h
        curx1 -= (invslope1 * (yval - h)) / Float32(h)
        curx2 -= (invslope2 * (yval - h)) / Float32(h)
        yval = h
    end
    while yval >= max(1, v2y(y(v1), h))
        for xval = max(1, u2x(curx1, w)+1):min(w, u2x(curx2, w))
            state = rasterfunc(xval, yval, image, state)
        end
        yval = yval - 1
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
function rastbot(image, shape::Triangle, state)
    w, h = size(image)

    v1, v2, v3 = vertices(shape)

    invslope1::Float32 = (x(v2) - x(v1)) / (y(v2) - y(v1))
    invslope2::Float32 = (x(v3) - x(v1)) / (y(v3) - y(v1))

    if invslope1 >= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = x(v1)
    curx2::Float32 = x(v1)

    yval = v2y(y(v1), h)
    if yval < 1
        curx1 -= (invslope1 * (1 - yval)) / Float32(h)
        curx2 -= (invslope2 * (1 - yval)) / Float32(h)
        yval = 1
    end
    while yval <= min(h, v2y(y(v2), h))
        for xval = max(1, u2x(curx1, w)+1):min(w, u2x(curx2, w))
            state = rasterfunc(xval, yval, image, state)
        end
        yval = yval + 1
        curx1 += invslope1 / Float32(h)
        curx2 += invslope2 / Float32(h)
    end

    state
end

function rasterize(image, tri::Triangle, state, ::RasterAlgorithmScanline)
    v1, v2, v3 = vertices(tri)

    # sort vertices in ascending order
    if y(v1) >= y(v2)
        v1, v2 = v2, v1
    end

    if y(v1) >= y(v3)
        v1, v3 = v3, v1
    end

    if y(v2) >= y(v3)
        v2, v3 = v3, v2
    end

    # Save work if the triangle is already flat
    if y(v2) == y(v3)
        state = rastbot(image, Triangle(v1, v2, v3), state)
    elseif y(v1) == y(v2)
        state = rasttop(image, Triangle(v1, v2, v3), state)
    else
        x4 = x(v1) + ((y(v2) - y(v1)) / (y(v3) - y(v1))) * (x(v3) - x(v1))
        v4 = Point(x4, y(v2))
        state = rastbot(image, Triangle(v1, v2, v4), state)
        state = rasttop(image, Triangle(v2, v4, v3), state)
    end
    state
end

end