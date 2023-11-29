module Spatial2D

export contains

using ..Shapes2D

contains(::AbstractShape, u, v) = error("Not implemented")

function contains(p::Polygon{N, T}, point::Pair{T, T}) where {N, T}
    c::Bool = false
    j::Int = N
    testx, testy = point
    for i = 1:N
        vix, viy = getvertex(p, i)
        vjx, vjy = getvertex(p, j)
        if ((viy>testy) != (vjy>testy)) && (testx < (vjx-vix) * (testy-viy) / (vjy-viy) + vix)
              c = !c;
        end

        j = i
    end

    c
end

end