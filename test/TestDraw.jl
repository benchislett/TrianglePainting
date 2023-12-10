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

        tri = Triangle{Float32}(initial)
        col = averagecolor(target, tri)
        loss = baseloss + drawloss(target, img, tri, col)

        if loss < best_x
            savevec_x = initial
            best_x = loss
        end
    end

    savevec_x, best_x
end

function getloss_gpu!(tris, rs, gs, bs, amounts, losses, target_d, img_d, NTris)
    tid::Int = threadIdx().x
    bid::Int = blockIdx().x
    gid::Int = (bid - 1) * blockDim().x + tid

    i, j = bid, tid
    w, h = size(target_d)

    u, v = Draw2D.uv(Float32, i, j, w, h)
    for tri_idx = 1:NTris
        tri::Triangle{Float32} = tris[tri_idx]
        if Spatial2D.contains(tri, Pair(u, v))
            @inbounds CUDA.atomic_add!(pointer(rs, tri_idx), target_d[i, j].r)
            @inbounds CUDA.atomic_add!(pointer(gs, tri_idx), target_d[i, j].g)
            @inbounds CUDA.atomic_add!(pointer(bs, tri_idx), target_d[i, j].b)
            CUDA.atomic_add!(pointer(amounts, tri_idx), UInt32(1))
        end
    end

    for tri_idx = 1:NTris
        tri::Triangle{Float32} = tris[tri_idx]
        if Spatial2D.contains(tri, Pair(u, v))
            col::RGB{Float32} = RGB{Float32}(rs[tri_idx], gs[tri_idx], bs[tri_idx]) / Float32(amounts[tri_idx])
            lossdiff::Float32 = Draw2D.absdiff(col, target_d[i, j]) - Draw2D.absdiff(img_d[i, j], target_d[i, j])
            CUDA.atomic_add!(pointer(losses, tri_idx), lossdiff)
        end
    end

    return
end

function run_gpu(N::Int, best_x, target, img)
    curngs = CUDA.rand(SVector{6,Float32}, N)
    target_gpu = cu(target)
    img_gpu = cu(img)
    device_losses = CUDA.zeros(Float32, N)
    device_rs = CUDA.zeros(Float32, N)
    device_gs = CUDA.zeros(Float32, N)
    device_bs = CUDA.zeros(Float32, N)
    device_sums = CUDA.zeros(UInt32, N)

    blobs_arr = map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, curngs))
    device_arr = map(Triangle{Float32}, blobs_arr)
    CUDA.@sync begin @cuda threads=200 blocks=200 getloss_gpu!(device_arr, device_rs, device_gs, device_bs, device_sums, device_losses, target_gpu, img_gpu, N) end
    minloss, minidx = findmin(device_losses)

    CUDA.@allowscalar blobs_arr[minidx], best_x + minloss
end

function refine(N::Int, savevec, best, baseloss, target, img)
    for j = 1:N
        initial::MVector{6,Float32} = savevec .+ (randn(Float32, 6) * 0.1)

        tri = Triangle{Float32}(initial)
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
    device_rs = CUDA.zeros(Float32, N)
    device_gs = CUDA.zeros(Float32, N)
    device_bs = CUDA.zeros(Float32, N)
    device_sums = CUDA.zeros(UInt32, N)

    blobs_arr = map(SVector{6,Float32}, map(vec -> savevec + (vec .- 0.5f0) .* 0.05f0, curngs))
    device_arr = map(Triangle{Float32}, blobs_arr)
    CUDA.@sync begin @cuda threads=200 blocks=200 getloss_gpu!(device_arr, device_rs, device_gs, device_bs, device_sums, device_losses, target_gpu, img_gpu, N) end
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
    cols = []

    @time for i = 1:100
        prevloss = imloss(target, img)

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

        tri = Triangle{Float32}(savevec)
        col = averagecolor(target, tri)

        if best < prevloss
            draw!(img, tri, col)
            draw!(img_big, tri, col)
            frame(anim, plot(img))
            push!(tris, tri)
            push!(cols, col)
        end

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
    
        for i = 1:200
            for j = 1:200
                u, v = Draw2D.uv(Float32, i, j, 200, 200)
                for k = length(tris):-1:1
                    if Spatial2D.contains(tris[k], Pair(u, v))
                        img[i, j] = colors[k] / Float32(quants[k])
                        break
                    end
                end
            end
        end 
    end

    println(imloss(target, img))

    save("output.png", img_big)
    save("output_orig.png", img)
    save("difference.png", abs.(img - target))

    gif(anim, "output.gif")

    return tris, cols
end

# main(1000000, 10)

# target = float.(load("lisa.png"))
# img = zeros(RGB{Float32}, 200, 200)

# initial = MVector{6,Float32}([rand(), rand(), rand(), rand(), rand(), rand()])
# tri = Triangle{Float32}(initial)
# col = averagecolor(target, tri)

# drawloss(target, img, tri, col)
# N = 10000
# rngs = rand(N, 6)
# # @profview run(N, 9999999.0f0, target, img, rngs)
# @btime run($N, 9999999.0f0, $target, $img, $rngs)
