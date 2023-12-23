module Pixel

using Images

export over
export Loss, SELoss, AELoss, loss

"""
    over(source, background)

Overlays an opaque pixel atop a background pixel.
Returns the source pixel.
"""
over(source::RGB{Float32}, background::RGB{Float32}) = source

"""
    over(source, background)

Overlays an alpha-transparent pixel atop an alpha-transparent background pixel.
Applies the over-compositing algorithm to the source pixel "over" the background.
"""
function over(source::RGBA{Float32}, background::RGBA{Float32})
    alpha_new = source.alpha + background.alpha * (1.0f0 - source.alpha)
    
    pixel_r = (source.r * source.alpha + background.alpha * background.r * (1.0f0 - source.alpha)) / alpha_new;
    pixel_g = (source.g * source.alpha + background.alpha * background.g * (1.0f0 - source.alpha)) / alpha_new;
    pixel_b = (source.b * source.alpha + background.alpha * background.b * (1.0f0 - source.alpha)) / alpha_new;

    RGBA{Float32}(pixel_r, pixel_g, pixel_b, alpha_new)
end

over(source::RGBA{Float32}, background::RGB{Float32}) = over(source, RGBA{Float32}(background))

"""
Loss type to use when comparing pixels.

One of [`SELoss`](@ref), [`AELoss`](@ref)
"""
abstract type Loss end

"""Squared Error pixel loss"""
struct SELoss <: Loss end

"""Absolute Error pixel loss"""
struct AELoss <: Loss end

"""
    loss(source, target, loss)

Return the loss between a source pixel and a target pixel, according to the provided Loss instance.

See also [`Loss`](@ref)
"""
loss(source::RGB{Float32}, target::RGB{Float32}, ::Loss) = error("Not implemented")

function loss(source::RGB{Float32}, target::RGB{Float32}, ::SELoss)::Float32
    dr::Float32 = source.r - target.r
    dg::Float32 = source.g - target.g
    db::Float32 = source.b - target.b
    return (dr * dr) + (dg * dg) + (db * db)
end

function loss(source::RGB{Float32}, target::RGB{Float32}, ::AELoss)::Float32
    deltapixel = abs.(source - target)
    deltapixel.r + deltapixel.g + deltapixel.b
end

function loss(source::RGB{Float32}, target::RGBA{Float32}, losstype::LossT) where LossT <: Loss
    loss(source, RGB{Float32}(target), losstype)
end

function loss(source::RGBA{Float32}, target::RGBA{Float32}, losstype::LossT) where LossT <: Loss
    loss(RGB{Float32}(source), RGB{Float32}(target), losstype)
end

function loss(source::RGBA{Float32}, target::RGB{Float32}, losstype::LossT) where LossT <: Loss
    loss(RGB{Float32}(source), target, losstype)
end

end