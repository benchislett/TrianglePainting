module GreedySim

using StaticArrays
using Images
using ImageFeatures
using IntervalSets
using Combinatorics
using StatsBase
using Serialization
using Random

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

const EvoShape = Polygon{4}

mutable struct SimState{Shape, Pixel}
    background::RGB{Float32}
    shapes::Vector{Shape}
    colours::Vector{Pixel}
    alpha::Float32
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
        draw!(state.current, state.shapes[k], RGBA{Float32}(state.colours[k].r, state.colours[k].g, state.colours[k].b, state.alpha), RasterAlgorithmBounded())
    end
end

function commit!(hist, state, target, shape, colour ; losstype, applyrecolor=true, applyglobalrefine=true)
    push!(state.shapes, shape)
    push!(state.colours, copy(colour))

    makecol(x) = RGBA{Float32}(x.r, x.g, x.b, state.alpha)

    if applyglobalrefine
        for round = 1:100
            Threads.@threads for which = 1:length(state.shapes)
                background = zero(target) .+ state.background
                for i = 1:(which-1)
                    draw!(background, state.shapes[i], makecol(state.colours[i]), RasterAlgorithmBounded())
                end
                
                foreground = zeros(RGBA{Float32}, size(target))
                for i = (which+1):length(state.shapes)
                    draw!(foreground, state.shapes[i], makecol(state.colours[i]), RasterAlgorithmBounded())
                end
                
                best = 0.0f0
                delta = zeros(Float32, numvars(eltype(state.shapes)))
                bestshape = state.shapes[which]
                for k = 1:1000
                    newshape = mutate(bestshape, delta .* 0.005f0)
                    ld = drawloss(target, background, newshape, makecol(state.colours[which]), SELoss(), RasterAlgorithmBounded(), foreground = foreground)
                    if ld < best
                        best = ld
                        bestshape = newshape
                    end
                    randn!(delta)
                end
            
                state.shapes[which] = bestshape
            end
        end
    end

    if applyrecolor
        if state.alpha >= 0.995
            state.colours, state.background = opaquerecolor(target, state.shapes, RasterAlgorithmBounded())
        else
            state.colours, state.background = alpharecolor(target, state.shapes, state.alpha, RasterAlgorithmBounded())
        end
        redraw!(state, target)
    else
        draw!(state.current, shape, colour, RasterAlgorithmBounded())
    end

    state.best = imloss(target, state.current, losstype)

    push!(hist.history, deepcopy(state))

    return
end

function simulate_iter_ga(state, target, tris, nbatch, nepochs, nrefinement, alpha ; losstype)
    raster_algorithm = RasterAlgorithmBounded()
    colours = averagepixel_batch(target, state.current, alpha, tris, raster_algorithm)
    losses = drawloss_batch(target, state.current, tris, colours, losstype, raster_algorithm)

    for roundidx = 1:nepochs
        for k=1:nrefinement
            rngs = randn(Float32, nbatch, 8) * range(0.025f0, 0.005f0, length=nrefinement)[k]
            newtris = mutate_batch(tris, rngs)
            newcolours = averagepixel_batch(target, state.current, alpha, newtris, raster_algorithm)
            newlosses = drawloss_batch(target, state.current, newtris, newcolours, losstype, raster_algorithm)
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

    minloss, mintri, clamp01.(mincol)
end

function simulate(target, nprims, nbatch, nepochs, nrefinement, alpha ; losstype = SELoss(), verbose=true)
    N = trunc(Int, floor(nbatch / 2))

    hist = SimLog{EvoShape, eltype(target)}([])

    state = SimState{EvoShape, eltype(target)}(averagepixel(target), [], [], alpha, zero(target), Inf32)
    # state = SimState{EvoShape, eltype(target)}(one(RGB{Float32}), [], [], alpha, zero(target), Inf32)
    redraw!(state, target)
    state.best = imloss(target, state.current, losstype)

    push!(hist.history, deepcopy(state))

    prev::Float32 = 0.0f0

    for primidx = 1:nprims
        itertime = @elapsed begin
            # function sample_prob(image ; N, scale_factor = 5.0)
            #     points = sample(1:prod(size(image)), Weights(reshape(Float32.(image).^scale_factor, prod(size(image)))), N)
            #     return collect(map(linearidx -> CartesianIndices(image)[linearidx].I, points))
            # end
            
            # diff = Gray.(abs.(state.current .- target))
            # diff = diff ./ maximum(diff)
            
            # points = sample_prob(diff, N = searchsortedfirst([binomial(i, 3) for i = 1:10000], N))
            # points = map(p -> Point(p[1] / size(target)[1], p[2] / size(target)[2]), points)
            # newtris = collect(map(Triangle, (combinations(points, 3))))[1:N]

            # rngs = 2.0f0 .* rand(Float32, N, 8) .- 0.5f0
            rngs = rand(Float32, N, 8)
            tris = [EvoShape(SVector{4, Point}(Point(slice[1], slice[2]), Point(slice[3], slice[4]), Point(slice[5], slice[6]), Point(slice[7], slice[8]))) for slice in eachslice(rngs, dims=1)]

            # tris = [firsttris ; newtris]
        
            minloss, mintri, mincol = simulate_iter_ga(state, target, tris, length(tris), nepochs, nrefinement, alpha, losstype=losstype)

            # function sample_loss(xs)
            #     tri = Triangle(SVector{6, Float32}(xs))
            #     col = averagepixel(target, tri, RasterAlgorithmBounded())
            #     loss = drawloss(target, state.current, tri, col, SELoss(), RasterAlgorithmBounded())
            #     Float64(loss)
            # end

            # optres = bboptimize(x -> sample_loss(x); SearchRange=(0, 1), TraceMode=:silent, NumDimensions=6, MaxTime=15, PopulationSize=2000)
            # if optres.archive_output.best_fitness < minloss
            #     println("BBOPTIMIZE useful")
            #     minloss = optres.archive_output.best_fitness
            #     mintri = Triangle(SVector{6, Float32}(optres.archive_output.best_candidate))
            #     mincol = averagepixel(target, mintri, RasterAlgorithmBounded())
            # end

            prev = state.best
            if minloss < 0 # normalized losses are negative if they reduce total loss
                commit!(hist, state, target, mintri, mincol, losstype=losstype)
            end
        end

        if verbose
            println("Added primitive $primidx with total loss ", state.best, " with difference ", prev - state.best, " in $itertime seconds")
        end

        Serialization.serialize("output/simresult/simlog_$nprims-prims_$nbatch-batch_$nepochs-epoch_$nrefinement-refine.bin", hist)
    end

    hist
end

end