module Paint
using Reexport

include("Shapes.jl")
@reexport using .Shapes2D

include("Spatial.jl")
@reexport using .Spatial2D

include("Pixel.jl")
@reexport using .Pixel

include("Raster.jl")
@reexport using .Raster2D

include("Draw.jl")
@reexport using .Draw2D

include("Mutate.jl")
@reexport using .Mutate

include("GreedySim.jl")
@reexport using .GreedySim

end # module Paint
