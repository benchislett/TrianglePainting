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
    alpha_new = source.a + background.a * (1.0f0 - source.a)
    
    pixel_r = (source.r * source.a + background.a * background.r * (1.0f0 - source.a)) / alpha_new;
    pixel_g = (source.g * source.a + background.a * background.g * (1.0f0 - source.a)) / alpha_new;
    pixel_b = (source.b * source.a + background.a * background.b * (1.0f0 - source.a)) / alpha_new;

    RGBA{Float32}(pixel_r, pixel_g, pixel_b, alpha_new)
end

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

loss(source::RGB{Float32}, target::RGB{Float32}, ::SELoss)::Float32 = abs2(source - target)

function loss(source::RGB{Float32}, target::RGB{Float32}, ::AELoss)::Float32
    deltapixel = abs.(source - target)
    deltapixel.r + deltapixel.g + deltapixel.b
end

end