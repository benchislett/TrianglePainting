import Test: @test, @testset
using ImageView, Images, Plots, FileIO
using StaticArrays
using BenchmarkTools
using Evolutionary
using Paint.Shapes2D
using Paint.Draw2D

import Random
Random.seed!(1234)

function run(N, best_x, target, img)
    f()::Float32 = 2 * rand(Float32) - 0.5

    baseloss = copy(best_x)
    savevec_x = MVector{6,Float32}([f(), f(), f(), f(), f(), f()])

    for j = 1:N
        initial = MVector{6,Float32}([f(), f(), f(), f(), f(), f()])

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

function refine(N, savevec, best, baseloss, target, img)
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
    img = zeros(RGBA{Float32}, 200, 200)
    img_big = zeros(RGBA{Float32}, 1024, 1024)

    f()::Float32 = 2 * rand(Float32) - 0.5

    nsplit = 1000

    prevloss = imloss(target, img)

    @time for i = 1:1000
        prevloss = imloss(target, img)

        best = copy(prevloss)
        savevec = MVector{6,Float32}([f(), f(), f(), f(), f(), f()])

        tasks = []
        for jj = 1:nsplit
            t = Threads.@spawn begin
                currvec, currbest = run(floor(N / nsplit), copy(prevloss), $target, $img)
            end

            # schedule(t)
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

        tasks = []
        for jj = 1:nsplit
            t = Threads.@spawn begin
                currvec, currbest = refine(floor(N / nsplit), $savevec, $best, $prevloss, $target, $img)
            end

            push!(tasks, t)
        end

        for jj = 1:nsplit
            vectmp, btmp = fetch(tasks[jj])
            if btmp < best
                best = btmp
                savevec = vectmp
            end
        end

        println(i, " ", best)

        tri = Triangle{Float32}(savevec)
        col = averagecolor(target, tri)

        draw!(img, tri, col)
        draw!(img_big, tri, col)
        frame(anim, plot(img))
    end

    println(imloss(target, img))

    save("output.png", img_big)

    gif(anim, "output.gif")
end

main(1)
main(100000)