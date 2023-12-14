module Draw2D

using Images

export draw!
export imloss, drawloss, averagepixel

using ..Spatial2D
using ..Shapes2D
using ..Pixel
using ..Raster2D

"""State for draw!"""
struct DrawRasterState{Pix}
    colour::Pix
end

"""Raster procedure for draw!"""
function rasterfunc(i, j, image, state::DrawRasterState{Pix}) where Pix
    @inbounds image[i, j] = over(state.colour, image[i, j])
    state
end

"""
    draw!(shape, image, colour, algorithm)

Draw a shape onto an image, using the given colour, according to the given rasterization algorithm.

See also [`RasterAlgorithm`](@ref)
"""
function draw!(shape, image, colour, algorithm = RasterAlgorithmPointwise())
    state = DrawRasterState(colour)
    rasterize(shape, image, state, algorithm)
    return
end

"""State for drawloss"""
struct DrawlossRasterState{Arr, Pix, LossType <: Loss}
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
    drawloss(shape, target, background, colour, loss, algorithm)

Return the total loss if a shape were drawn an image, using the given colour, according to the given rasterization algorithm.

See also [`Loss`](@ref), [`RasterAlgorithm`](@ref)
"""
function drawloss(shape, target, background, colour, lossstate, algorithm = RasterAlgorithmPointwise())
    state = DrawlossRasterState(target, colour, lossstate, 0.0f0)
    state = rasterize(shape, background, state, algorithm)
    state.total
end

"""State for averagepixel"""
struct PixelAverageRasterState{Pix}
    colour::Pix
    count::Int32
end

"""Raster procedure for averagepixel"""
function rasterfunc(i, j, image, state::PixelAverageRasterState{Pix}) where Pix
    @inbounds PixelAverageRasterState{Pix}(state.colour + image[i, j], state.count + 1)
end

"""
    averagepixel(shape, image, algorithm)

Return the average pixel value of pixels in an image which are contained in a shape.

See also [`RasterAlgorithm`](@ref)
"""
function averagepixel(shape, image, algorithm = RasterAlgorithmPointwise())
    state = PixelAverageRasterState{eltype(image)}(zero(eltype(image)), 0)
    state = rast(shape, image, state, alg)
    state.colour / state.count
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
