module Spatial2D

export contains
export points

using ..Shapes2D

contains(::AbstractShape, u, v) = error("Not implemented")

function contains(p::Polygon{N,T}, point::Pair{T,T}) where {N,T}
    c::Bool = false
    j::Int = N
    testx, testy = point
    for i = 1:N
        vix, viy = getvertex(p, i)
        vjx, vjy = getvertex(p, j)
        if ((viy > testy) != (vjy > testy)) && (testx < (vjx - vix) * (testy - viy) / (vjy - viy) + vix)
            c = !c
        end

        j = i
    end

    c
end

struct RasterizedSet{S<:AbstractShape}
    shape::S
    width::Int
    height::Int
end

struct RasterState
    i::Int
    j::Int
end

import Base.iterate

function iterate(r::RasterizedSet, st::RasterState=RasterState(1, 1))
    nextstate = RasterState(st.i + 1, st.j)
    if st.i == r.width
        nextstate = RasterState(1, st.j + 1)
    end

    if st.j > r.height
        return nothing
    end

    u = (Float32(st.i) - 0.5f0) / Float32(r.width)
    v = (Float32(st.j) - 0.5f0) / Float32(r.height)

    if contains(r.shape, Pair(u, v))
        return (Pair(st.i, st.j), nextstate)
    else
        return iterate(r, nextstate)
    end
end

function points(shape, width::Int, height::Int)
    ptsiterator = (Pair(i, j) for i in 1:width for j in 1:height)
    pttouv(ij) = Pair((Float32(ij.first) - 0.5f0) / Float32(width), (Float32(ij.second) - 0.5f0) / Float32(height))

    Iterators.filter(pt -> contains(shape, pttouv(pt)), ptsiterator)
end

end