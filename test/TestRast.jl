import Test: @test, @testset
using ImageView, Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint

function shape()
    v1, v2, v3 = Point(0.25, 0.1), Point(0.55, 0.75), Point(0.65, 0.35)
    shape = Triangle(v1, v2, v3)
end

function draw_and_save(colour, algorithm, testname)
    image = ones(typeof(colour), 200, 200)
    draw!(image, shape(), colour, algorithm)
    save("output/raster_tests/" * testname * ".png", image)
    return true
end

function main()
    opaque_colour = RGB{Float32}(0.25, 0.25, 0.55)
    draw_and_save(opaque_colour, RasterAlgorithmPointwise(), "opaque_pointwise")
    draw_and_save(opaque_colour, RasterAlgorithmBounded(), "opaque_bounded")
    draw_and_save(opaque_colour, RasterAlgorithmScanline(), "opaque_scanline")

    transparent_colour = RGBA{Float32}(0.25, 0.25, 0.55, 0.5)
    draw_and_save(transparent_colour, RasterAlgorithmPointwise(), "transparent_pointwise")
    draw_and_save(transparent_colour, RasterAlgorithmBounded(), "transparent_bounded")
    draw_and_save(transparent_colour, RasterAlgorithmScanline(), "transparent_scanline")
end

main()
