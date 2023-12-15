module GreedySim

using StaticArrays

using ..Shapes2D
using ..Raster2D
using ..Draw2D
using ..Pixel

export PrimitiveSequence, SimState
export simulate

mutable struct SimState{Shape, Pixel}
    shapes::Vector{Shape}
    current_colours::Vector{Pixel}
    original_colours::Vector{Pixel}
    current::Array{Pixel, 2}
    best::Float32
end

function simulate(target, nprims, nbatch, losstype = SELoss())
    state = SimState{Triangle, eltype(target)}([], [], [], zero(target) .+ one(eltype(target)), Inf32)
    state.best = imloss(target, state.current, losstype)

    for primidx = 1:nprims
        rngs = 2.0f0 .* rand(Float32, nbatch, 6) .- 0.5f0 # todo generalize random initialization
        tris = [Triangle(SVector{6, Float32}(slice)) for slice in eachslice(rngs, dims=1)]
        colours = averagepixel_batch(target, tris, RasterAlgorithmScanline())
        losses = drawloss_batch(target, state.current, tris, colours, losstype, RasterAlgorithmScanline())

        minloss, minidx = findmin(losses)
        if minloss < 0 # normalized losses are negative if they reduce total loss
            push!(state.shapes, tris[minidx])
            push!(state.current_colours, colours[minidx])
            push!(state.original_colours, colours[minidx])

            draw!(state.current, tris[minidx], colours[minidx], RasterAlgorithmScanline())
            println("Added primitive $primidx with total loss ", state.best + minloss, " and delta $minloss")
        end

        state.best = imloss(target, state.current, losstype)
    end

    state
end

end