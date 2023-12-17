module GreedySim

using StaticArrays
using Images

using ..Shapes2D
using ..Raster2D
using ..Spatial2D
using ..Draw2D
using ..Pixel
using ..Mutate

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
    w, h = size(target)
    initial = zero(target) .+ one(eltype(target))
    state = SimState{Triangle, eltype(target)}([], [], [], initial, Inf32)
    state.best = imloss(target, state.current, losstype)

    for primidx = 1:nprims
        rngs = 2.0f0 .* rand(Float32, nbatch, 6) .- 0.5f0 # todo generalize random initialization
        tris = [Triangle(SVector{6, Float32}(slice)) for slice in eachslice(rngs, dims=1)]
        colours = averagepixel_batch(target, tris, RasterAlgorithmScanline())
        losses = drawloss_batch(target, state.current, tris, colours, losstype, RasterAlgorithmScanline())

        minloss = 0.0f0
        minidx = 0
        for roundidx = 1:5
            for k=1:50
                rngs = randn(Float32, nbatch * 100) * 0.1f0
                newtris = mutate_batch(tris, rngs)
                newcolours = averagepixel_batch(target, newtris, RasterAlgorithmScanline())
                newlosses = drawloss_batch(target, state.current, newtris, newcolours, losstype, RasterAlgorithmScanline())
                Threads.@threads for i=1:nbatch
                    if newlosses[i] < losses[i]
                        losses[i] = newlosses[i]
                        tris[i] = newtris[i]
                        colours[i] = newcolours[i]
                    end
                end
            end

            minloss, minidx = findmin(losses)
            fill!(tris, tris[minidx])
            fill!(colours, colours[minidx])
            fill!(losses, minloss)
        end

        if minloss < 0 # normalized losses are negative if they reduce total loss

            push!(state.shapes, tris[minidx])
            push!(state.original_colours, copy(colours[minidx]))
            push!(state.current_colours, copy(colours[minidx]))

            # recolor
            recolors = zeros(RGB{Float32}, length(state.shapes))
            quants = zeros(UInt32, length(state.shapes))

            for i = 1:w
                for j = 1:h
                    pt = Point(x2u(i, w), y2v(j, h))
                    for k = length(state.shapes):-1:1
                        if covers(state.shapes[k], pt)
                            recolors[k] += target[i, j]
                            quants[k] += 1
                            break
                        end
                    end
                end
            end

            for k = length(state.shapes):-1:1
                state.current_colours[k] = recolors[k] / Float32(quants[k])
            end

            initial = zero(target) .+ one(eltype(target))
            state.current = initial
            for k=1:length(state.shapes)
                draw!(state.current, state.shapes[k], state.current_colours[k], RasterAlgorithmScanline())
            end

            prev = state.best

            state.best = imloss(target, state.current, losstype)

            println("Added primitive $primidx with total loss ", state.best, " with difference ", prev - state.best)
        end

        state.best = imloss(target, state.current, losstype)
    end

    state
end

end