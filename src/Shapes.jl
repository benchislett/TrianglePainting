module Shapes2D

import Base.eltype

export AbstractShape, AbstractPolygon
export Polygon
export Triangle

export getvertex
export eltype

using StaticArrays

abstract type AbstractShape end

abstract type AbstractPolygon <: AbstractShape end

struct Polygon{N,T<:AbstractFloat} <: AbstractPolygon
    vertices::SVector{N,Pair{T,T}}
end

eltype(::Polygon{N,T}) where {N,T} = T

getvertex(p::Polygon, i) = getindex(p.vertices, i)
min(p::Polygon) = reduce((x, y) -> Pair(Base.min(x.first, y.first), Base.min(x.second, y.second)), p.vertices)
max(p::Polygon) = reduce((x, y) -> Pair(Base.max(x.first, y.first), Base.max(x.second, y.second)), p.vertices)

const Triangle{T} = Polygon{3,T}

Triangle{T}(v1, v2, v3) where {T} = Triangle{T}(SVector{3,Pair{T,T}}(v1, v2, v3))
Triangle{T}(vertices::SVector{6,T}) where {T} = Triangle{T}(Pair(vertices[1], vertices[2]), Pair(vertices[3], vertices[4]), Pair(vertices[5], vertices[6]))
Triangle{T}(vertices::MVector{6,T}) where {T} = Triangle{T}(Pair(vertices[1], vertices[2]), Pair(vertices[3], vertices[4]), Pair(vertices[5], vertices[6]))

end