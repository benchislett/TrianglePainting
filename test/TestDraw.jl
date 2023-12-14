import Test: @test, @testset
using Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint.Shapes2D
using Paint.Draw2D
using Paint.Spatial2D

using CUDA

import Random
Random.seed!(1234)

ENV["GKSwstype"]="nul"

function run(N::Int, best_x, target, img)
    baseloss = copy(best_x)
    savevec_x = MVector{6,Float32}([0, 0, 0, 0, 0, 0])

    rngs = 2 .* rand(Float32, N, 6) .- 0.5

    for j = 1:N
        initial = MVector{6,Float32}(view(rngs, j, :))

        tri = Triangle(initial)
        col = averagecolor(target, tri)
        loss = baseloss + drawloss(target, img, tri, col)

        if loss < best_x
            savevec_x = initial
            best_x = loss
        end
    end

    savevec_x, best_x
end


# Only returns from lane 0
function warpreducesum(amt)
    mask = CUDA.active_mask()
    amt += CUDA.shfl_down_sync(mask, amt, 16)
    amt += CUDA.shfl_down_sync(mask, amt, 8)
    amt += CUDA.shfl_down_sync(mask, amt, 4)
    amt += CUDA.shfl_down_sync(mask, amt, 2)
    amt += CUDA.shfl_down_sync(mask, amt, 1)
    return amt
end

function warpbroadcast(val)
    return CUDA.shfl_sync(CUDA.active_mask(), val, 1)
end

toxcoord(x, w) = Int(floor(w * x))
toycoord(x, h) = Int(floor(h * x))

function getloss_gpu!(tris, losses, target_d, img_d, NTris)
    @inbounds begin
        tid::Int = threadIdx().x
        bid::Int = blockIdx().x

        shape = tris[bid]

        minx, miny = Shapes2D.min(shape)
        maxx, maxy = Shapes2D.max(shape)

        w, h = size(target_d)

        col::RGB{Float32} = zero(RGB{Float32})
        amt::UInt32 = 0

        xfloor = max(1, toxcoord(minx, w)) - 1
        xceil = min(w, toxcoord(maxx, w) + 1) - 1
        for xbase = xfloor:32:xceil
            x = xbase + tid
            if x <= w
                for y in max(1, toycoord(miny, h)):min(h, toycoord(maxy, h) + 1)
                    u, v = Draw2D.uv(eltype(shape), x, y, w, h)
                    if Spatial2D.contains(shape, Pair(u, v))
                        col += target_d[x, y]
                        amt += 1
                    end
                end
            end
        end

        amtwarp = warpreducesum(amt)
        colr = warpreducesum(col.r)
        colg = warpreducesum(col.g)
        colb = warpreducesum(col.b)

        amt = warpbroadcast(amtwarp)
        colr = warpbroadcast(colr)
        colg = warpbroadcast(colg)
        colb = warpbroadcast(colb)

        col = RGB{Float32}(colr, colg, colb)

        colour::RGB{Float32} = col / Float32(amt)

        loss::Float32 = 0.0f0

        for xbase = xfloor:32:xceil
            x = xbase + tid
            if x <= w
                for y in max(1, toycoord(miny, h)):min(h, toycoord(maxy, h) + 1)
                    u, v = Draw2D.uv(eltype(shape), x, y, w, h)
                    if Spatial2D.contains(shape, Pair(u, v))
                        loss += Draw2D.absdiff(colour, target_d[x, y]) - Draw2D.absdiff(img_d[x, y], target_d[x, y])
                    end
                end
            end
        end

        loss = warpreducesum(loss)

        if tid == 1
            losses[bid] = loss
        end

        return
    end
end

function run_gpu(N::Int, best_x, target, img)
    curngs = CUDA.rand(SVector{6,Float32}, N)
    target_gpu = cu(target)
    img_gpu = cu(img)
    device_losses = CUDA.zeros(Float32, N)

    blobs_arr = map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, curngs))
    device_arr = map(Triangle, blobs_arr)
    CUDA.@sync begin @cuda threads=32 blocks=N getloss_gpu!(device_arr, device_losses, target_gpu, img_gpu, N) end
    minloss, minidx = findmin(device_losses)

    CUDA.@allowscalar blobs_arr[minidx], best_x + minloss
end

function refine(N::Int, savevec, best, baseloss, target, img)
    for j = 1:N
        initial::MVector{6,Float32} = savevec .+ (randn(Float32, 6) * 0.1)

        tri = Triangle(initial)
        col = averagecolor(target, tri)
        loss = baseloss + drawloss(target, img, tri, col)

        if loss < best
            savevec = initial
            best = loss
        end
    end

    savevec, best
end

function refine_gpu(N::Int, savevec, best_x, baseloss, target, img)
    curngs = CUDA.rand(SVector{6,Float32}, N)
    target_gpu = cu(target)
    img_gpu = cu(img)
    device_losses = CUDA.zeros(Float32, N)

    blobs_arr = map(SVector{6,Float32}, map(vec -> savevec + (vec .- 0.5f0) .* 0.05f0, curngs))
    device_arr = map(Triangle, blobs_arr)
    CUDA.@sync begin @cuda threads=32 blocks=N getloss_gpu!(device_arr, device_losses, target_gpu, img_gpu, N) end
    minloss, minidx = findmin(device_losses)

    CUDA.@allowscalar blobs_arr[minidx], baseloss + minloss
end

function main(N, nsplit)

    anim = Animation()

    target = float.(load("lisa.png"))
    img = zeros(RGB{Float32}, 200, 200)
    img_big = zeros(RGB{Float32}, 1024, 1024)

    f()::Float32 = 2 * rand(Float32) - 0.5

    prevloss = imloss(target, img)
    
    tris = []
    cols_orig = []
    cols = []
    losses = []

    @time for i = 1:100
        prevloss = imloss(target, img)
        push!(losses, prevloss)

        best = copy(prevloss)
        savevec = MVector{6,Float32}([f(), f(), f(), f(), f(), f()])

        # for jj = 1:nsplit
        #     vectmp, btmp = run(Int(floor(N / nsplit)), copy(prevloss), target, img)
        #     if btmp < best
        #         best = btmp
        #         savevec = vectmp
        #     end
        # end

        tasks = []
        for jj = 1:nsplit
            t = Threads.@spawn begin
                vectmp, besttmp = run_gpu(Int(floor(N / nsplit)), copy(prevloss), target, img)
                for z=1:100
                    vectmp, besttmp = refine_gpu(Int(floor(N / nsplit)), vectmp, besttmp, copy(prevloss), target, img)
                end
                refine_gpu(Int(floor(N / nsplit)), vectmp, besttmp, copy(prevloss), target, img)
            end

            push!(tasks, t)
        end

        best = copy(prevloss)

        for jj = 1:nsplit
            vectmp, btmp = fetch(tasks[jj])
            if btmp < best
                best = btmp
                savevec = vectmp
            end
        end

        ## REFINE

        # tasks = []
        # for jj = 1:nsplit
        #     t = Threads.@spawn begin
        #         currvec, currbest = refine(Int(floor(N / nsplit)), savevec, best, prevloss, target, img)
        #     end

        #     push!(tasks, t)
        # end

        # for jj = 1:nsplit
        #     vectmp, btmp = fetch(tasks[jj])
        #     if btmp < best
        #         best = btmp
        #         savevec = vectmp
        #     end
        # end

        println(i, " ", best)

        tri = Triangle(savevec)
        col = averagecolor(target, tri)

        if best < prevloss
            draw!(img, tri, col)
            draw!(img_big, tri, col)
            frame(anim, plot(img))
            push!(tris, tri)
            push!(cols_orig, col)
            push!(cols, col)

            colors = zeros(RGB{Float32}, length(tris))
            quants = zeros(UInt32, length(tris))

            for i = 1:200
                for j = 1:200
                    u, v = Draw2D.uv(Float32, i, j, 200, 200)
                    for k = length(tris):-1:1
                        if Spatial2D.contains(tris[k], Pair(u, v))
                            colors[k] += target[i, j]
                            quants[k] += 1
                            break
                        end
                    end
                end
            end

            for k = length(tris):-1:1
                cols[k] = colors[k] / Float32(quants[k])
            end
        
            for i = 1:200
                for j = 1:200
                    u, v = Draw2D.uv(Float32, i, j, 200, 200)
                    for k = length(tris):-1:1
                        if Spatial2D.contains(tris[k], Pair(u, v))
                            img[i, j] = cols[k]
                            break
                        end
                    end
                end
            end
        end
    end

    prevloss = imloss(target, img)
    push!(losses, prevloss)

    save("output.png", img_big)
    save("output_orig.png", img)
    save("difference.png", abs.(img - target))

    gif(anim, "output.gif")

    return tris, cols, losses
end

# main(1000000, 10)

# target = float.(load("lisa.png"))
# img = zeros(RGB{Float32}, 200, 200)

# initial = MVector{6,Float32}([rand(), rand(), rand(), rand(), rand(), rand()])
# tri = Triangle(initial)
# col = averagecolor(target, tri)

# drawloss(target, img, tri, col)
# N = 10000
# rngs = rand(N, 6)
# # @profview run(N, 9999999.0f0, target, img, rngs)
# @btime run($N, 9999999.0f0, $target, $img, $rngs)
