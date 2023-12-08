using Images, FileIO
using StaticArrays
using BenchmarkTools
using Paint.Shapes2D
using Paint.Draw2D

using CUDA

function main()
    img = zeros(RGB{Float32}, 200, 200)
    target = float.(load("lisa.png"))

    img_gpu = cu(img)
    target_gpu = cu(target)

    baseloss::Float32 = imloss(target, img)

    function getloss(tri::Triangle{Float32})::Float32
        col::RGB{Float32} = averagecolor(target, tri, RastAlgorithmScanline())
        baseloss + drawloss(target, img, tri, col, RastAlgorithmScanline())
    end

    function getloss_gpu(tri::Triangle{Float32})::Float32
        col::RGB{Float32} = averagecolor(target_gpu, tri, RastAlgorithmPointwise())
        baseloss + drawloss(target_gpu, img_gpu, tri, col, RastAlgorithmPointwise())
    end

    N::Int = 1000000

    rngs = rand(SVector{6,Float32}, N)
    curngs = cu(rngs)
    
    host_arr = map(Triangle{Float32}, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, rngs)))
    host_losses = @time map(getloss, host_arr)
    host_min = findmin(host_losses)

    device_arr = CUDA.@sync map(Triangle{Float32}, map(SVector{6,Float32}, map(vec -> 2 .* vec .- 0.5, curngs)))
    device_losses = @time CUDA.@sync map(getloss_gpu, device_arr)
    device_min = findmin(device_losses)

    host_min, device_min
end

main()