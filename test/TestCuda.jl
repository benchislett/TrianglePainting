using Images, FileIO
using StaticArrays
using BenchmarkTools
using Paint.Shapes2D
using Paint.Draw2D
using Paint.Spatial2D

using CUDA

import Random
Random.seed!(1234)

function main()
    img = zeros(RGB{Float32}, 200, 200)
    target = float.(load("lisa.png"))

    img_gpu = cu(img)
    target_gpu = cu(target)

    baseloss::Float32 = imloss(target, img)

    N::Int = 10000

    function getloss(tri::Triangle{Float32})::Float32
        col::RGB{Float32} = averagecolor(target, tri, RastAlgorithmPointwise())
        drawloss(target, img, tri, col, RastAlgorithmPointwise())
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

    rngs = rand(SVector{6,Float32}, N)
    curngs = cu(rngs)
    device_losses = CUDA.zeros(Float32, N)
    device_rs = CUDA.zeros(Float32, N)
    device_gs = CUDA.zeros(Float32, N)
    device_bs = CUDA.zeros(Float32, N)
    device_sums = CUDA.zeros(UInt32, N)
    
    host_arr = map(Triangle{Float32}, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, rngs)))
    host_losses = @time map(getloss, host_arr)
    host_min = findmin(host_losses)

    device_arr = CUDA.@sync map(Triangle{Float32}, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, curngs)))
    @time CUDA.@sync begin @cuda threads=200 blocks=200 getloss_gpu!(device_arr, device_rs, device_gs, device_bs, device_sums, device_losses, target_gpu, img_gpu, N) end
    device_min = findmin(device_losses)

    # host_min, device_min
    device_min
end

main()