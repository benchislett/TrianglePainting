module Draw2D

using Images

export draw!
export imloss, drawloss, drawloss_batch, averagepixel, averagepixel_batch, opaquerecolor

using ..Spatial2D
using ..Shapes2D
using ..Pixel
using ..Raster2D

# Explicit import to specialize
import ..Raster2D: rasterfunc

"""State for draw!"""
struct DrawRasterState{Pix} <: RasterState
    colour::Pix
end

"""Raster procedure for draw!"""
function rasterfunc(i, j, image, state::DrawRasterState{Pix}) where Pix
    @inbounds image[i, j] = over(state.colour, image[i, j])
    state
end

"""
    draw!(image, shape, colour, algorithm)

Draw a shape onto an image, using the given colour, according to the given rasterization algorithm.

See also [`RasterAlgorithm`](@ref)
"""
function draw!(image, shape, colour, algorithm = RasterAlgorithmPointwise())
    state = DrawRasterState(colour)
    rasterize(image, shape, state, algorithm)
    return true
end

"""State for drawloss"""
struct DrawlossRasterState{Arr, Pix, LossType <: Loss} <: RasterState
    target::Arr
    colour::Pix
    loss::LossType
    total::Float32
end

"""Raster procedure for drawloss"""
function rasterfunc(i, j, image, state::DrawlossRasterState{Arr, Pix, LossType}) where {Arr, Pix, LossType}
    @inbounds newtotal::Float32 = state.total + loss(state.colour, state.target[i, j], state.loss) - loss(image[i, j], state.target[i, j], state.loss)
    DrawlossRasterState{Arr, Pix, LossType}(state.target, state.colour, state.loss, newtotal)
end

"""
    drawloss(target, background, shape, colour, loss, algorithm)

Return the total loss if a shape were drawn on an image, using the given colour, according to the given rasterization algorithm.

See also [`Loss`](@ref), [`RasterAlgorithm`](@ref)
"""
function drawloss(target, background, shape, colour, lossstate, algorithm = RasterAlgorithmPointwise())
    state = DrawlossRasterState(target, colour, lossstate, 0.0f0)
    state = rasterize(background, shape, state, algorithm)
    state.total
end

"""
    drawloss_batch(target, background, shapes, colours, loss, algorithm)

Return a vector of losses given a collection of shapes and colours, calculated in the same fashion as drawloss.

See also [`drawloss`](@ref)
"""
function drawloss_batch(target, background, shapes, colours, lossstate, algorithm = RasterAlgorithmPointwise())
    totals = Vector{Float32}(undef, length(shapes))
    Threads.@threads for i=1:length(shapes)
        @inbounds totals[i] = drawloss(target, background, shapes[i], colours[i], lossstate, algorithm)
    end
    totals
end

"""State for averagepixel"""
struct PixelAverageRasterState{Pix} <: RasterState
    colour::Pix
    count::Int32
end

"""Raster procedure for averagepixel"""
function rasterfunc(i, j, image, state::PixelAverageRasterState{Pix}) where Pix
    @inbounds PixelAverageRasterState{Pix}(state.colour + image[i, j], state.count + 1)
end

"""
    averagepixel(image, shape, algorithm)

Return the average pixel value of pixels in an image which are contained in a shape.

See also [`RasterAlgorithm`](@ref)
"""
function averagepixel(image, shape, algorithm = RasterAlgorithmPointwise())
    state = PixelAverageRasterState{eltype(image)}(zero(eltype(image)), 0)
    state = rasterize(image, shape, state, algorithm)
    state.colour / Float32(max(Int32(1), state.count))
end

"""
    averagepixel(image)

Return the average pixel value of all pixels in an image.
"""
function averagepixel(image)
    sum(image) / prod(size(image))
end

"""
    averagepixel_batch(target, shapes, algorithm)

Return a vector of average pixels in an image, calculated in the same fashion as averagepixel.

See also [`averagepixel`](@ref)
"""
function averagepixel_batch(target, shapes, algorithm = RasterAlgorithmPointwise())
    pixels = Vector{eltype(target)}(undef, length(shapes))
    Threads.@threads for i in eachindex(shapes)
        @inbounds pixels[i] = averagepixel(target, shapes[i], algorithm)
    end
    pixels
end

"""State for opaquerecolor"""
struct OpaqueRecolorRasterState <: RasterState
    visited::BitArray{2}
    colour::RGB{Float32}
    count::Int32
end

"""Raster procedure for opaquerecolor"""
function rasterfunc(i, j, image, state::OpaqueRecolorRasterState)
    if state.visited[i, j]
        state
    else
        state.visited[i, j] = 1
        return @inbounds OpaqueRecolorRasterState(state.visited, state.colour + image[i, j], state.count + 1)
    end
end

"""
    opaquerecolor(target, shapes, algorithm)

Return the optimal coloring assignment for a collection of overlapping, opaque shapes.

See also [`RasterAlgorithm`](@ref)
"""
function opaquerecolor(target, shapes, algorithm)
    colours = Vector{RGB{Float32}}(undef, length(shapes))
    visited = BitArray{2}(undef, size(target))
    fill!(visited, 0)
    for k = length(shapes):-1:1
        state = OpaqueRecolorRasterState(visited, zero(RGB{Float32}), 0)
        state = rasterize(target, shapes[k], state, algorithm)
        colours[k] = state.colour / Float32(state.count)
    end

    bg = zero(RGB{Float32})
    count = 0
    for i = 1:size(target)[1]
        for j = 1:size(target)[2]
            if !visited[i, j]
                bg += target[i, j]
                count += 1
            end
        end
    end

    colours, bg / Float32(max(count, 1))
end

"""
    imloss(target, background, loss)

Return the total loss between two images, according to a loss type.

See also [`Loss`](@ref)
"""
function imloss(target, background, lossstate)
    w, h = size(target)

    total::Float32 = 0.0f0

    for i = 1:w
        for j = 1:h
            @inbounds total += loss(target[i, j], background[i, j], lossstate)
        end
    end

    total
end

end
