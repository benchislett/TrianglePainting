using Paint
using Images

import Random
Random.seed!(1234)

function main()
    NTris = 10

    rm("output/simresult/", force=true, recursive=true)
    mkpath("output/simresult/")
    target = float.(load("lisa.png"))
    @time res = simulate(target, NTris, 1000)
    save("output/simresult/final.png", res.current)

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

main()