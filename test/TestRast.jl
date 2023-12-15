import Test: @test, @testset
using ImageView, Images, Plots, FileIO
using Paint

function triangle()
    v1, v2, v3 = Point(0.25, 0.1), Point(0.55, 0.75), Point(0.65, 0.35)
    Triangle(v1, v2, v3)
end

function rectangle()
    AABB(Point(0.15, 0.25), Point(0.75, 0.9))
end

function draw_and_save(shape, colour, algorithm, testname)
    image = ones(typeof(colour), 128, 128)
    @test draw!(image, shape, colour, algorithm)
    save("output/raster_tests/" * testname * ".png", image)
end

rm("output/raster_tests/", force=true, recursive=true)
mkpath("output/raster_tests/")

@testset "Rasterization" verbose=true begin
    @testset "Opaque" begin
        colour = RGB{Float32}(0.25, 0.25, 0.55)
        @testset "Triangle" begin
            shape = triangle()
            @testset "Pointwise" draw_and_save(shape, colour, RasterAlgorithmPointwise(), "opaque_triangle_pointwise")
            @testset "Bounded"   draw_and_save(shape, colour, RasterAlgorithmBounded(), "opaque_triangle_bounded")
            @testset "Scanline"  draw_and_save(shape, colour, RasterAlgorithmScanline(), "opaque_triangle_scanline")
        end
        
        @testset "Rectangle" begin
            shape = rectangle()
            @testset "Pointwise" draw_and_save(shape, colour, RasterAlgorithmPointwise(), "opaque_rectangle_pointwise")
            @testset "Bounded"   draw_and_save(shape, colour, RasterAlgorithmBounded(), "opaque_rectangle_bounded")
        end
    end

    @testset "Transparent" begin
        colour = RGBA{Float32}(0.25, 0.25, 0.55, 0.5)
        @testset "Triangle" begin
            shape = triangle()
            @testset "Pointwise" draw_and_save(shape, colour, RasterAlgorithmPointwise(), "transparent_triangle_pointwise")
            @testset "Bounded"   draw_and_save(shape, colour, RasterAlgorithmBounded(), "transparent_triangle_bounded")
            @testset "Scanline"  draw_and_save(shape, colour, RasterAlgorithmScanline(), "transparent_triangle_scanline")
        end

        @testset "Rectangle" begin
            shape = rectangle()
            @testset "Pointwise" draw_and_save(shape, colour, RasterAlgorithmPointwise(), "transparent_rectangle_pointwise")
            @testset "Bounded"   draw_and_save(shape, colour, RasterAlgorithmBounded(), "transparent_rectangle_bounded")
        end
    end
end

