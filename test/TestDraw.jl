import Test: @test, @testset
using ImageView, Images, Plots
using StaticArrays

using Paint.Shapes2D
using Paint.Draw2D

img = zeros(RGBA{Float32}, 200, 200)

for i = 1:50
    tri = Polygon{3, Float32}([Pair(rand(), rand()), Pair(rand(), rand()), Pair(rand(), rand())])
    col = RGBA{Float32}(rand(), rand(), rand(), 1)


    draw!(img, tri, col)
end

imshow(img)

