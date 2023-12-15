module Spatial2D

export covers

using ..Shapes2D

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
    j::Int = N
    testx, testy = point
    for i = 1:N
        vix, viy = vertex(p, i)
        vjx, vjy = vertex(p, j)
        if ((viy > testy) != (vjy > testy)) && (testx < (vjx - vix) * (testy - viy) / (vjy - viy) + vix)
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