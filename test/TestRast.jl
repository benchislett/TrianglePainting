import Test: @test, @testset
using ImageView, Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint.Shapes2D
using Paint.Draw2D

import Random
Random.seed!(1234)

f() = rand() * 2 - 0.5
v1, v2, v3 = Pair(f(), f()), Pair(f(), f()), Pair(f(), f())
shape = Triangle{Float32}(v1, v2, v3)
shape2 = Triangle{Float64}(v1, v2, v3)
# count2::Int = 0
# @btime begin
#     for i in 1:100
#         rast(shape2, 200, 200, (i, j, u, v) -> begin
#             global count2
#             count2 = count2 + Int(floor(10 * u))
#         end)
#     end
# end
# println(count2)
# count::Int = 0
# @btime begin
#     for i in 1:100
#         rast(shape, 200, 200, (i, j, u, v) -> begin
#             global count
#             count = count + Int(floor(10 * u))
#         end)
#     end
# end
# println(count)

target = float.(load("lisa.png"))
img = zeros(RGB{Float32}, 200, 200)
img2 = zeros(RGB{Float64}, 200, 200)

col = averagecolor(target, shape)

# @btime drawloss($target, $img, $shape2, $col)
# @btime drawloss($target, $img, $shape, $col)

draw!(img, shape, col)

col2 = averagecolor(target, shape2)
draw!(img2, shape2, col2)

save("output.png", img)
save("output2.png", img2)
