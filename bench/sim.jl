using Paint
using Images
using BenchmarkTools

function main()
    target = float.(load("lisa.png"))
    NTris = 1
    NBatch = 10000
    NEpoch = 1
    NRefine = 10
    res = @benchmark simulate($target, $NTris, $NBatch, $NEpoch, $NRefine, verbose=false)
    
    fastest_run_ns = minimum(res.times)
    num_evals = NTris * NBatch * (1 + (NEpoch * NRefine))
    evals_per_second = float(num_evals) / (fastest_run_ns / 1000000000.0)
    println("Evals per second: $evals_per_second")

    return res
end

res = main()