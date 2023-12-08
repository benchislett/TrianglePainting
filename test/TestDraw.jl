import Test: @test, @testset
using Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint.Shapes2D
using Paint.Draw2D

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

function main(N)

    anim = Animation()

    target = float.(load("lisa.png"))
    img = zeros(RGB{Float32}, 200, 200)
    img_big = zeros(RGB{Float32}, 1024, 1024)

    f()::Float32 = 2 * rand(Float32) - 0.5

    nsplit = 1

    prevloss = imloss(target, img)

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
                vectmp, besttmp = run(Int(floor(N / nsplit)), copy(prevloss), target, img)
                # refine(Int(floor(N / nsplit)), vectmp, besttmp, copy(prevloss), target, img)
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
        end
    end

    println(imloss(target, img))

    save("output.png", img_big)
    save("difference.png", abs.(img - target))

    gif(anim, "output.gif")

    img
end

main(100000)

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