module GreedySim

using StaticArrays
using Images

using Evolutionary

using ..Shapes2D
using ..Raster2D
using ..Spatial2D
using ..Draw2D
using ..Pixel
using ..Mutate

export PrimitiveSequence, SimState
export simulate, commit!, redraw!, genbackground

mutable struct SimState{Shape, Pixel}
    background::RGB{Float32}
    shapes::Vector{Shape}
    current_colours::Vector{Pixel}
    original_colours::Vector{Pixel}
    current::Array{Pixel, 2}
    best::Float32
end

function redraw!(state, target)
    initial = zero(target) .+ state.background
    state.current = initial
    for k in eachindex(state.shapes)
        draw!(state.current, state.shapes[k], state.current_colours[k], RasterAlgorithmScanline())
    end
end

function commit!(state, target, shape, colour ; losstype, applyrecolor=true)
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
end

function simulate_iter_ga(state, target, nbatch, nepochs, nrefinement ; losstype)
    rngs = 2.0f0 .* rand(Float32, nbatch, 6) .- 0.5f0 # todo generalize random initialization
    tris = [Triangle(SVector{6, Float32}(slice)) for slice in eachslice(rngs, dims=1)]
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
        upper = Int(floor(nbatch/10))
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
    state = SimState{Triangle, eltype(target)}(averagepixel(target), [], [], [], zero(target), Inf32)
    redraw!(state, target)
    state.best = imloss(target, state.current, losstype)

    for primidx = 1:nprims
        minloss, mintri, mincol = simulate_iter_ga(state, target, nbatch, nepochs, nrefinement, losstype=losstype)

        """
        function sample_loss(xs)
            tri = Triangle(xs)
            col = averagepixel(target, tri, RasterAlgorithmScanline())
            loss = drawloss(target, state.current, tri, col, losstype, RasterAlgorithmScanline())
        end

        init_sample = [rand(Float32, 6) for i=1:100]
        optres = Evolutionary.optimize(
            sample_loss,
            Evolutionary.BoxConstraints([0.0f0 for i = 1:6], [1.0f0 for i=1:6]),
            init_sample,
            Evolutionary.CMAES(mu=100, metrics=Vector{Evolutionary.ConvergenceMetric}([])),
            Evolutionary.Options(iterations=1000)
        )
        println(optres)
        minloss = optres.minimum
        mintri = Triangle(SVector{6, Float32}(optres.minimizer))
        mincol = averagepixel(target, mintri, RasterAlgorithmScanline())
        # minloss, minidx = findmin(losses)
        # mintri = tris[minidx]
        # mincol = colours[minidx]
        """

        if minloss < 0 # normalized losses are negative if they reduce total loss
            prev = state.best
            commit!(state, target, mintri, mincol, losstype=losstype)

            if verbose
                println("Added primitive $primidx with total loss ", state.best, " with difference ", prev - state.best)
            end
        end
    end

    state
end

end