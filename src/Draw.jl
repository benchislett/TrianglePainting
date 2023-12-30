module Draw2D

using Images

export draw!
export imloss, drawloss, drawloss_batch, averagepixel, averagepixel_batch, opaquerecolor, alpharecolor

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
    nextcol = over(state.colour, image[i, j])
    @inbounds newtotal::Float32 = state.total + loss(nextcol, state.target[i, j], state.loss) - loss(image[i, j], state.target[i, j], state.loss)
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
    Threads.@threads for i=eachindex(shapes)
        @inbounds totals[i] = drawloss(target, background, shapes[i], colours[i], lossstate, algorithm)
    end
    totals
end

"""State for averagepixel"""
struct PixelAverageRasterState <: RasterState
    current::Array{RGB{Float32}, 2}
    colour::RGB{Float32}
    alpha::Float32
    count::Int32
end

"""Raster procedure for averagepixel"""
function rasterfunc(i, j, image, state::PixelAverageRasterState)
    @inbounds PixelAverageRasterState(state.current, state.colour + ((image[i, j] - ((1.0f0 - state.alpha) .* state.current[i, j])) ./ state.alpha), state.alpha, state.count + 1)
end

"""
    averagepixel(target, current, shape, algorithm)

Return the average pixel value of pixels in an image which are contained in a shape.

See also [`RasterAlgorithm`](@ref)
"""
function averagepixel(target, current, alpha, shape, algorithm = RasterAlgorithmPointwise())
    state = PixelAverageRasterState(current, zero(RGB{Float32}), alpha, 0)
    state = rasterize(target, shape, state, algorithm)
    col = state.colour / Float32(max(Int32(1), state.count))
    RGBA{Float32}(col.r, col.g, col.b, alpha)
end

"""
    averagepixel(image)

Return the average pixel value of all pixels in an image.
"""
function averagepixel(image)
    sum(image) / prod(size(image))
end

"""
    averagepixel_batch(target, current, alpha, shapes, algorithm)

Return a vector of average pixels in an image, calculated in the same fashion as averagepixel.

See also [`averagepixel`](@ref)
"""
function averagepixel_batch(target, current, alpha, shapes, algorithm = RasterAlgorithmPointwise())
    pixels = Vector{RGBA{Float32}}(undef, length(shapes))
    Threads.@threads for i in eachindex(shapes)
        @inbounds pixels[i] = averagepixel(target, current, alpha, shapes[i], algorithm)
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
    @inbounds begin
        if state.visited[i, j]
            state
        else
            state.visited[i, j] = 1
            return OpaqueRecolorRasterState(state.visited, state.colour + image[i, j], state.count + 1)
        end
    end
end

"""
    opaquerecolor(target, shapes, algorithm)

Return the optimal coloring assignment for a collection of overlapping, opaque shapes.

See also [`RasterAlgorithm`](@ref)
"""
function opaquerecolor(target, shapes, algorithm)
    @inbounds begin
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
end

"""State for alpharecolor"""
struct AlphaRecolorRasterState <: RasterState
    A::Matrix{Float64}
    coeffs::Matrix{Float64}
    alpha::Float64
    k::Int32
end

function rasterfunc(i, j, image, state::AlphaRecolorRasterState)
    @inbounds begin
        colidx = 3 * (state.k - 1) + 1
        t_idx = ((j - 1) * size(image)[1] + (i - 1)) + 1
        a_idx = (t_idx - 1) * 3 + 1
        coeff = state.coeffs[i, j]

        state.A[a_idx + 0, colidx + 0] = coeff
        state.A[a_idx + 1, colidx + 1] = coeff
        state.A[a_idx + 2, colidx + 2] = coeff

        state.coeffs[i, j] = coeff * (1.0f0 - state.alpha)
    end

    state
end

function alpharecolor(target, shapes, alpha, algorithm)
    A = zeros(3 * prod(size(target)), 3 * length(shapes) + 3)
    y = zeros(3 * prod(size(target)))
    coeffs = Array{Float64, 2}(undef, size(target))
    fill!(coeffs, alpha)

    for k = length(shapes):-1:1
        state = AlphaRecolorRasterState(A, coeffs, alpha, k)
        rasterize(target, shapes[k], state, algorithm)
    end

    w, h = size(target)
    @inbounds for j = 1:h
        for i = 1:w
            t_idx = ((j - 1) * size(target)[1] + (i - 1)) + 1
            a_idx = (t_idx - 1) * 3 + 1

            y[a_idx + 0] = target[t_idx].r
            y[a_idx + 1] = target[t_idx].g
            y[a_idx + 2] = target[t_idx].b

            A[a_idx + 0, 3 * length(shapes) + 1] = coeffs[i, j] / alpha
            A[a_idx + 1, 3 * length(shapes) + 2] = coeffs[i, j] / alpha
            A[a_idx + 2, 3 * length(shapes) + 3] = coeffs[i, j] / alpha
        end
    end

    x = A \ y

    colours = Vector{RGB{Float32}}(undef, length(shapes))
    for k in eachindex(shapes)
        colidx = 3 * (k - 1) + 1
        @inbounds colours[k] = clamp01.(RGB{Float32}(x[colidx + 0], x[colidx + 1], x[colidx + 2]))
    end

    background = clamp01.(RGB{Float32}(x[3 * length(shapes) + 1], x[3 * length(shapes) + 2], x[3 * length(shapes) + 3]))

    colours, background
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
