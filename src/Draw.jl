module Draw2D

export draw!

using ..Spatial2D
using ..Shapes2D

function draw!(img, shape, colour)
    w, h = size(img)

    for i = 1:w
        for j = 1:h
            u::Float32 = Float32(i-1) / Float32(w)
            v::Float32 = Float32(j-1) / Float32(h)

            if Spatial2D.contains(shape, Pair(u, v))
                img[i, j] = colour
            end
        end
    end
end


end