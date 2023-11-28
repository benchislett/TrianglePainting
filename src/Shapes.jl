module Shapes2D

export AbstractShape, AbstractPolygon
export Polygon
# export Triangle

export getvertex

using StaticArrays

abstract type AbstractShape end

abstract type AbstractPolygon <: AbstractShape end

getvertex(::AbstractPolygon, ::Int) = error("Not implemented")

struct Polygon{N, T<:AbstractFloat}
    vertices::SVector{N, Pair{T, T}}
end

getvertex(p::Polygon, i) = getindex(p.vertices, i)

# const Triangle{T} = Polygon{3, T}

# Triangle{T}(v1, v2, v3) where T = Triangle{T}(SVector{3, Pair{T, T}}(v1, v2, v3))

end