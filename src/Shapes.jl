module Shapes2D

using StaticArrays

abstract type AbstractShape end

contains(::AbstractShape, u, v) = error("Not implemented")

abstract type AbstractPolygon <: AbstractShape end

getvertex(::AbstractPolygon, ::Int) = error("Not implemented")

struct Polygon{N::Int, T<:AbstractFloat}
    vertices::SVector{N, T}
end

getvertex(p::Polygon{N, T}, i) = getindex(p.vertices, i)

abstract type AbstractTriangle <: AbstractPolygon end

v1(::AbstractTriangle) = error("Not implemented")
v2(::AbstractTriangle) = error("Not implemented")
v3(::AbstractTriangle) = error("Not implemented")

function getvertex(t::AbstractTriangle, i::Int)
    if i == 1
        v1(t)
    elseif i == 2
        v2(t)
    elseif i == 3
        v3(t)
    else
        throw(BoundsError(t, i))
    end
end

struct Triangle{T<:AbstractFloat}
    vertices::SVector{3, T}
end

getvertex(t::Triangle{T}, i) = getindex(t.vertices, i)
v1(t::Triangle{T}) = getindex(t.vertices, 1)
v2(t::Triangle{T}) = getindex(t.vertices, 2)
v3(t::Triangle{T}) = getindex(t.vertices, 3)

end