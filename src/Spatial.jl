module Spatial2D

export covers

using ..Shapes2D

using LLVM

"""
    covers(shape, point)

Check if a point is contained within a shape.
"""
covers(::AbstractShape, ::Point) = error("Not implemented")

"""
    covers(polygon, point)

Generic point-in-polygon test.
Ported from https://wrfranklin.org/Research/Short_Notes/pnpoly.html
"""
function covers(p::Polygon{N}, point::Point) where {N}
    c::Bool = false
    j::Int32 = N
    testx, testy = point
    for i::Int32 = 1:N
        vix, viy = vertex(p, i)
        vjx, vjy = vertex(p, j)
        dy::Float32 = vjy - viy
        dx::Float32 = vjx - vix
        LLVM.Interop.assume(dy != 0.0)
        if ((viy > testy) != (vjy > testy)) && (testx < dx * (testy - viy) / dy + vix)
            c = !c
        end

        j = i
    end

    c
end

"""
    covers(aabb, point)

Check if a point is contained in a 2D bounding box.
"""
covers(aabb::AABB, point::Point) = (x(aabb.min) < x(point) < x(aabb.max)) && (y(aabb.min) < y(point) < y(aabb.max))

end