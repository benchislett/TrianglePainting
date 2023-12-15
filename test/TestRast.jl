import Test: @test, @testset
using ImageView, Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint

import Random
Random.seed!(1234)

function main()
    f() = rand() * 2 - 0.5
    v1, v2, v3 = Point(f(), f()), Point(f(), f()), Point(f(), f())
    shape = Triangle(v1, v2, v3)

    target::Array{RGB{Float32}, 2} = float.(load("lisa.png"))

    col = averagepixel(shape, target)

    img = zeros(RGB{Float32}, 200, 200)
    draw!(shape, img, col, RasterAlgorithmPointwise())
    save("output_pointwise.png", img)

    img = zeros(RGB{Float32}, 200, 200)
    draw!(shape, img, col, RasterAlgorithmBounded())
    save("output_bounded.png", img)

    img = zeros(RGB{Float32}, 200, 200)
    draw!(shape, img, col, RasterAlgorithmScanline())
    save("output_scanline.png", img)
end

main()
