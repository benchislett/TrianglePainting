module GreedySim

using StaticArrays
using Images
using ImageFeatures
using IntervalSets
using Combinatorics
using StatsBase
using Serialization

using Evolutionary
using BlackBoxOptim

using ..Shapes2D
using ..Raster2D
using ..Spatial2D
using ..Draw2D
using ..Pixel
using ..Mutate

export PrimitiveSequence, SimState
export simulate, commit!, redraw!, genbackground, simulate_iter_ga

mutable struct SimState{Shape, Pixel}
    background::RGB{Float32}
    shapes::Vector{Shape}
    current_colours::Vector{Pixel}
    original_colours::Vector{Pixel}
    current::Array{Pixel, 2}
    best::Float32
end

mutable struct SimLog{Shape, Pixel}
    history::Vector{SimState{Shape, Pixel}}
end

function redraw!(state, target)
    initial = zero(target) .+ state.background
    state.current = initial
    for k in eachindex(state.shapes)
        draw!(state.current, state.shapes[k], state.current_colours[k], RasterAlgorithmScanline())
    end
end

function commit!(hist, state, target, shape, colour ; losstype, applyrecolor=true)
    push!(state.shapes, shape)
    push!(state.original_colours, copy(colour))
    push!(state.current_colours, copy(colour))

    if applyrecolor
        state.current_colours, state.background = opaquerecolor(target, state.shapes, RasterAlgorithmScanline())
        redraw!(state, target)
    else
        draw!(state.current, shape, colour, RasterAlgorithmScanline())
    end

    state.best = imloss(target, state.current, losstype)

    push!(hist.history, deepcopy(state))

    return
end

function simulate_iter_ga(state, target, tris, nbatch, nepochs, nrefinement ; losstype)
    colours = averagepixel_batch(target, tris, RasterAlgorithmScanline())
    losses = drawloss_batch(target, state.current, tris, colours, losstype, RasterAlgorithmScanline())

    for roundidx = 1:nepochs
        for k=1:nrefinement
            rngs = randn(Float32, nbatch, 6) * 0.05f0
            newtris = mutate_batch(tris, rngs)
            newcolours = averagepixel_batch(target, newtris, RasterAlgorithmScanline())
            newlosses = drawloss_batch(target, state.current, newtris, newcolours, losstype, RasterAlgorithmScanline())
            for i=1:nbatch
                if newlosses[i] < losses[i]
                    losses[i] = newlosses[i]
                    tris[i] = newtris[i]
                    colours[i] = newcolours[i]
                end
            end
        end

        idxs = sortperm(losses)
        upper = Int32(floor(nbatch/10))
        toptris = copy(tris[idxs[1:upper]])
        topcols = copy(colours[idxs[1:upper]])
        toploss = copy(losses[idxs[1:upper]])

        Threads.@threads for i=0:upper-1
            for k=1:10
                tris[10 * i + k] = toptris[i + 1]
                colours[10 * i + k] = topcols[i + 1]
                losses[10 * i + k] = toploss[i + 1]
            end
        end
    end

    minloss, minidx = findmin(losses)
    mintri = tris[minidx]
    mincol = colours[minidx]

    minloss, mintri, mincol
end

function simulate(target, nprims, nbatch, nepochs, nrefinement ; losstype = SELoss(), verbose=true)
    hist = SimLog{Triangle, eltype(target)}([])

    state = SimState{Triangle, eltype(target)}(averagepixel(target), [], [], [], zero(target), Inf32)
    redraw!(state, target)
    state.best = imloss(target, state.current, losstype)

    push!(hist.history, deepcopy(state))

    for primidx = 1:nprims
        function sample_prob(image ; N, scale_factor = 5.0)
            points = sample(1:prod(size(image)), Weights(reshape(Float32.(image).^scale_factor, prod(size(image)))), N)
            return collect(map(linearidx -> CartesianIndices(image)[linearidx].I, points))
        end

        diff = Gray.(abs.(state.current .- target))
        diff = diff ./ maximum(diff)

        points = sample_prob(diff, N = 100)
        points = map(p -> Point(p[1] / 200.0, p[2] / 200.0), points)
        newtris = collect(map(Triangle, (combinations(points, 3))))

        rngs = 2.0f0 .* rand(Float32, nbatch, 6) .- 0.5f0
        firsttris = [Triangle(SVector{6, Float32}(slice)) for slice in eachslice(rngs, dims=1)]

        tris = [firsttris ; newtris]
    
        minloss, mintri, mincol = simulate_iter_ga(state, target, tris, length(tris), nepochs, nrefinement, losstype=losstype)

        function sample_loss(xs)
            tri = Triangle(SVector{6, Float32}(xs))
            col = averagepixel(target, tri, RasterAlgorithmScanline())
            loss = drawloss(target, state.current, tri, col, SELoss(), RasterAlgorithmScanline())
            Float64(loss)
        end

        optres = bboptimize(x -> sample_loss(x); SearchRange=(0, 1), TraceMode=:silent, NumDimensions=6, MaxTime=15, PopulationSize=2000)
        if optres.archive_output.best_fitness < minloss
            minloss = optres.archive_output.best_fitness
            mintri = Triangle(SVector{6, Float32}(optres.archive_output.best_candidate))
            mincol = averagepixel(target, mintri, RasterAlgorithmScanline())
        end

        if minloss < 0 # normalized losses are negative if they reduce total loss
            prev = state.best
            commit!(hist, state, target, mintri, mincol, losstype=losstype)

            if verbose
                println("Added primitive $primidx with total loss ", state.best, " with difference ", prev - state.best)
            end
            Serialization.serialize("output/simresult/simlog_$nprims-prims_$nbatch-batch_$nepochs-epoch_$nrefinement-refine.bin", hist)
        end
    end

    hist
end

end