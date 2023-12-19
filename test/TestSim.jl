using Paint
using Images
using Serialization
import Random
Random.seed!(1234)

function main()
    NTris = 100
    NBatch = 100
    NEpoch = 5
    NRefine = 100

    # rm("output/simresult/", force=true, recursive=true)
    mkpath("output/simresult/")
    target = float.(load("lisa.png"))
    @time res = simulate(target, NTris, NBatch, NEpoch, NRefine)
    # save("output/simresult/final.png", res.current)
    # Serialization.serialize("output/simresult/simres_$NTris-prims_$NBatch-batch_$NEpoch-epoch_$NRefine-refine.bin", res)

    # img_acc = ones(RGB{Float32}, 200, 200)
    # for i=1:NTris
    #     img = ones(RGB{Float32}, 200, 200)
    #     draw!(img, res.shapes[i], res.original_colours[i], RasterAlgorithmScanline())
    #     draw!(img_acc, res.shapes[i], res.original_colours[i], RasterAlgorithmScanline())
    #     save("output/simresult_single$i.png", img)
    #     save("output/simresult_accum$i.png", img_acc)
    # end

    res
end

res = main()