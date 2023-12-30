module Paint

include("Shapes.jl")
using .Shapes2D
export AbstractShape, AbstractPolygon
export Point, x, y
export AABB
export Polygon, Triangle
export vertices, vertex

include("Spatial.jl")
using .Spatial2D
export covers

include("Pixel.jl")
using .Pixel
export over
export Loss, SELoss, AELoss, loss

include("Raster.jl")
using .Raster2D
export u2x, v2y, x2u, y2v
export RasterState, rasterfunc, rasterize
export RasterAlgorithm, RasterAlgorithmScanline, RasterAlgorithmBounded, RasterAlgorithmPointwise

include("Draw.jl")
using .Draw2D
export draw!
export imloss, drawloss, drawloss_batch, averagepixel, averagepixel_batch, opaquerecolor, alpharecolor

include("Mutate.jl")
using .Mutate
export numvars, mutate, mutate_batch

include("DrawGPU.jl")
using .GPUDraw2D
export RasterAlgorithmGPU

include("GreedySim.jl")
using .GreedySim
export PrimitiveSequence, SimState
export simulate, commit!, redraw!, genbackground, simulate_iter_ga

end # module Paint
