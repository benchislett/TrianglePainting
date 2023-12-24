module Shapes2D

export AbstractShape, AbstractPolygon
export Point, x, y
export AABB
export Polygon, Triangle
export vertices, vertex

using StaticArrays

"""Point in 2D"""
const Point = Pair{Float32, Float32}
"""Return the x coordinate of a 2D point"""
x(p::Point) = first(p)
"""Return the y coordinate of a 2D point"""
y(p::Point) = last(p)

abstract type AbstractShape end
abstract type AbstractPolygon <: AbstractShape end

"""Polygon in 2D with N vertices"""
struct Polygon{N} <: AbstractPolygon
    vertices::SVector{N, Point}
end

"""Return the static array of vertices of a Polygon"""
vertices(p::Polygon) = p.vertices
"""Return the Point of the i-th vertex of a Polygon"""
vertex(p::Polygon, i) = @inbounds p.vertices[i]

"""Axis-Aligned Bounding Box in 2D"""
struct AABB <: AbstractShape
    min::Point
    max::Point
end

AABB(s::AbstractShape) = AABB(Point(0.0f0, 0.0f0), Point(1.0f0, 1.0f0))
function AABB(p::Polygon)
    minx = reduce(min, (x(p) for p in p.vertices))
    miny = reduce(min, (y(p) for p in p.vertices))
    maxx = reduce(max, (x(p) for p in p.vertices))
    maxy = reduce(max, (y(p) for p in p.vertices))
    AABB(Point(minx, miny), Point(maxx, maxy))
end
AABB(points::SVector{4,Float32}) = @inbounds AABB(Point(points[1], points[2]), Point(points[3], points[4]))
AABB(points::MVector{4,Float32}) = @inbounds AABB(Point(points[1], points[2]), Point(points[3], points[4]))
AABB(points::Vector{Float32}) = @inbounds AABB(Point(points[1], points[2]), Point(points[3], points[4]))

"""Triangle in 2D, alias for a 3-vertex Polygon"""
const Triangle = Polygon{3}

Triangle(v1, v2, v3) = Triangle(SVector{3,Point}(v1, v2, v3))
Triangle(vertices::SVector{6,Float32}) = @inbounds Triangle(Point(vertices[1], vertices[2]), Point(vertices[3], vertices[4]), Point(vertices[5], vertices[6]))
Triangle(vertices::MVector{6,Float32}) = @inbounds Triangle(Point(vertices[1], vertices[2]), Point(vertices[3], vertices[4]), Point(vertices[5], vertices[6]))
Triangle(vertices::Vector{Float32}) = @inbounds Triangle(Point(vertices[1], vertices[2]), Point(vertices[3], vertices[4]), Point(vertices[5], vertices[6]))

end