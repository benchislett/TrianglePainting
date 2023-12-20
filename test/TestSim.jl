using Paint
using Images
using Serialization
import Random
Random.seed!(1234)

function main()
    NTris = 100
    NBatch = 5000
    NEpoch = 5
    NRefine = 250

    # rm("output/simresult/", force=true, recursive=true)
    mkpath("output/simresult/")
    target = float.(load("lisa.png"))
    @time res = simulate(target, NTris, NBatch, NEpoch, NRefine)
    save("output/simresult/experiment.png", res.history[end].current)
    Serialization.serialize("output/simresult/simlog_$NTris-prims_10x-blackbox_$NBatch-batch_$NEpoch-epoch_$NRefine-refine.bin", res)

    res
end

res = main()