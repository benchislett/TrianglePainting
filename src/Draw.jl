module Draw2D

using Images

export draw!
export imloss, drawloss, averagecolor

using ..Spatial2D
using ..Shapes2D

absdiff(r1, r2) = abs(r1.r - r2.r) + abs(r1.g - r2.g) + abs(r1.b - r2.b)

function shaperange(shape, w, h)
    tocoord(x) = Int(floor(w * x))
    minx, miny = Shapes2D.min(shape)
    maxx, maxy = Shapes2D.max(shape)

    return ((i, j) for i = max(1, tocoord(minx)):min(w, tocoord(maxx) + 1) for j = max(1, tocoord(miny)):min(h, tocoord(maxy) + 1))
end

uv(i, j, w, h) = (Float32(i) - 0.5f0) / Float32(w), (Float32(j) - 0.5f0) / Float32(h)

function draw!(img, shape, colour)
    w, h = size(img)

    for (i, j) in shaperange(shape, w, h)
        u, v = uv(i, j, w, h)
        if Spatial2D.contains(shape, Pair(u, v))
            img[i, j] = colour
        end
    end

end

function drawloss(target, background, shape, colour)
    w, h = size(target)

    total::Float32 = 0.0f0
    for (i, j) in shaperange(shape, w, h)
        u, v = uv(i, j, w, h)
        if Spatial2D.contains(shape, Pair(u, v))
            total += absdiff(colour, target[i, j])
            total -= absdiff(background[i, j], target[i, j])
        end
    end

    total
end

function averagecolor(target, shape)
    w, h = size(target)

    col = RGBA{Float32}(0, 0, 0, 0)
    n::Int = 0
    for (i, j) in shaperange(shape, w, h)
        u, v = uv(i, j, w, h)
        if Spatial2D.contains(shape, Pair(u, v))
            col = col + target[i, j]
            n = n + 1
        end
    end

    col / n
end

function imloss(img1, img2)
    w, h = size(img1)

    total::Float32 = 0.0f0

    for i = 1:w
        for j = 1:h
            @inbounds total += absdiff(img1[i, j], img2[i, j])
        end
    end

    total
end

end