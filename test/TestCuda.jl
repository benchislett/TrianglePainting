using Images, FileIO
using StaticArrays
using BenchmarkTools
using Paint.Shapes2D
using Paint.Draw2D
using Paint.Spatial2D

using CUDA

import Random
Random.seed!(1234)

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

function main()
    img = zeros(RGB{Float32}, 200, 200)
    target = float.(load("lisa.png"))

    img_gpu = cu(img)
    target_gpu = cu(target)

    baseloss::Float32 = imloss(target, img)

    N::Int = 10000

    function getloss(tri::Triangle)::Float32
        col::RGB{Float32} = averagecolor(target, tri, RastAlgorithmPointwise())
        drawloss(target, img, tri, col, RastAlgorithmPointwise())
    end

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

    rngs = rand(SVector{6,Float32}, N)
    curngs = cu(rngs)
    device_losses = CUDA.zeros(Float32, N)
    
    host_arr = map(Triangle, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, rngs)))
    host_losses = @time map(getloss, host_arr)
    host_min = findmin(host_losses)

    device_arr = CUDA.@sync map(Triangle, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, curngs)))
    @time CUDA.@sync begin @cuda threads=32 blocks=N getloss_gpu!(device_arr, device_losses, target_gpu, img_gpu, N) end
    device_min = findmin(device_losses)

    host_min, device_min
    # device_min
end

main()