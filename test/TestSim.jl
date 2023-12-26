using Paint
using Images
using Serialization
import Random
Random.seed!(1234)

function main()
    NTris = 100
    NBatch = 1000000
    NEpoch = 10
    NRefine = 100

    # rm("output/simresult/", force=true, recursive=true)
    mkpath("output/simresult/")
    target = float.(load("lisa.png"))
    @time res = simulate(target, NTris, NBatch, NEpoch, NRefine, 0.75f0)
    save("output/simresult/experiment.png", res.history[end].current)

    res
end

res = main()
