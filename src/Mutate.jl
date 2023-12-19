module Mutate

using ..Shapes2D

export numvars, mutate, mutate_batch

"""
    numvars(shapetype)

Return the number of parameters needed to initialize or mutate a shape.
"""
numvars(::Type{AbstractShape}) = error("Not implemented")
numvars(::Type{AABB}) = 4
numvars(::Type{Polygon{N}}) where {N} = 2 * N

"""
    mutate(shape, nums)

Return a mutated shape, parameterized by `numvars(typeof(shape))` floats.

See also [`numvars`](@ref)
"""
mutate(::AbstractShape, nums) = error("Not implemented")

mutate(aabb::AABB, nums) = @inbounds AABB(Point(x(aabb.min) + nums[1], y(aabb.min) + nums[2]), Point(x(aabb.max) + nums[3], y(aabb.max) + nums[4]))

function mutate(tri::Triangle, nums)
    @inbounds Triangle(Point(x(vertex(tri, 1)) + nums[1], y(vertex(tri, 1)) + nums[2]),
                       Point(x(vertex(tri, 2)) + nums[3], y(vertex(tri, 2)) + nums[4]),
                       Point(x(vertex(tri, 3)) + nums[5], y(vertex(tri, 3)) + nums[6]))
end

"""
    mutate_batch(shapes, nums)

Return a vector of mutated shapes, parameterized by `length(shapes) * numvars(eltype(shapes))` floats.

See also [`numvars`](@ref)
"""
function mutate_batch(shapes::Vector{T}, nums) where T <: AbstractShape
    newshapes = Vector{T}(undef, length(shapes))
    Threads.@threads for i = 1:length(shapes)
        offset = (i - 1) * numvars(T) + 1
        @inbounds newshapes[i] = mutate(shapes[i], nums[offset:offset + numvars(T) - 1])
    end
    newshapes
end

end

