module Draw2D

using Images

export draw!
export imloss, drawloss, averagecolor
export rast

using ..Spatial2D
using ..Shapes2D

absdiff(r1, r2) = abs(r1.r - r2.r) + abs(r1.g - r2.g) + abs(r1.b - r2.b)

function uv(::Type{T}, i, j, w, h) where {T<:Real}
    (T(i) - T(0.5)) / T(w), (T(j) - T(0.5)) / T(h)
end
toxcoord(x, w) = Int(floor(w * x))
toycoord(x, h) = Int(floor(h * x))

function shaperange(shape, w, h)
    minx, miny = Shapes2D.min(shape)
    maxx, maxy = Shapes2D.max(shape)

    return ((i, j) for i = max(1, toxcoord(minx, w)):min(w, toxcoord(maxx, w) + 1) for j = max(1, toycoord(miny, h)):min(h, toycoord(maxy, h) + 1))
end

# http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
function rasttop(shape::Triangle{Float32}, w, h, f)
    v1x, v1y = getvertex(shape, 1)
    v2x, v2y = getvertex(shape, 2)
    v3x, v3y = getvertex(shape, 3)

    invslope1::Float32 = (v3x - v1x) / (v3y - v1y)
    invslope2::Float32 = (v3x - v2x) / (v3y - v2y)

    if invslope1 <= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = v3x
    curx2::Float32 = v3x

    y::Int = toycoord(v3y, h)
    if y > h
        curx1 -= (invslope1 * (y - h)) / Float32(h)
        curx2 -= (invslope2 * (y - h)) / Float32(h)
        y = h
    end
    while y >= max(1, toycoord(v1y, h))
        for x = max(1, toxcoord(curx1, w)):min(w, toxcoord(curx2, w))
            u, v = uv(Float32, x, y, w, h)
            f(x, y, u, v)
        end
        y = y - 1
        curx1 -= invslope1 / Float32(h)
        curx2 -= invslope2 / Float32(h)
    end
end

function rastbot(shape::Triangle{Float32}, w, h, f)
    v1x, v1y = getvertex(shape, 1)
    v2x, v2y = getvertex(shape, 2)
    v3x, v3y = getvertex(shape, 3)

    invslope1::Float32 = (v2x - v1x) / (v2y - v1y)
    invslope2::Float32 = (v3x - v1x) / (v3y - v1y)

    if invslope1 >= invslope2
        invslope1, invslope2 = invslope2, invslope1
    end

    curx1::Float32 = v1x
    curx2::Float32 = v1x

    y::Int = toycoord(v1y, h)
    if y < 1
        curx1 -= (invslope1 * (1 - y)) / Float32(h)
        curx2 -= (invslope2 * (1 - y)) / Float32(h)
        y = 1
    end
    while y <= min(h, toycoord(v2y, h))
        for x = max(1, toxcoord(curx1, w)):min(w, toxcoord(curx2, w))
            u, v = uv(Float32, x, y, w, h)
            f(x, y, u, v)
        end
        y = y + 1
        curx1 += invslope1 / Float32(h)
        curx2 += invslope2 / Float32(h)
    end
end

function rast(shape::Triangle{Float32}, w, h, f)
    v1, v2, v3 = shape.vertices

    # sort vertices ascending
    if v1.second >= v2.second
        v1, v2 = v2, v1
    end

    if v1.second >= v3.second
        v1, v3 = v3, v1
    end

    if v2.second >= v3.second
        v2, v3 = v3, v2
    end

    if v2.second == v3.second
        rastbot(Triangle{Float32}(v1, v2, v3), w, h, f)
    elseif v1.second == v2.second
        rasttop(Triangle{Float32}(v1, v2, v3), w, h, f)
    else
        x4 = v1.first + ((v2.second - v1.second) / (v3.second - v1.second)) * (v3.first - v1.first)
        v4 = Pair(x4, v2.second)
        rastbot(Triangle{Float32}(v1, v2, v4), w, h, f)
        rasttop(Triangle{Float32}(v2, v4, v3), w, h, f)
    end
end

function rast(shape, w, h, f)
    tocoord(x) = Int(floor(w * x))
    minx, miny = Shapes2D.min(shape)
    maxx, maxy = Shapes2D.max(shape)

    for y in max(1, tocoord(miny)):min(h, tocoord(maxy) + 1)
        for x in max(1, tocoord(minx)):min(w, tocoord(maxx) + 1)
            u, v = uv(eltype(shape), x, y, w, h)
            if Spatial2D.contains(shape, Pair(u, v))
                f(x, y, u, v)
            end
        end
    end
end

function draw!(img, shape, colour)
    w, h = size(img)

    let col = colour
        rast(shape, w, h, @inline((i, j, u, v) -> @inbounds img[i, j] = col))
    end
end

function drawloss(target, background, shape, colour)
    w, h = size(target)

    total::Float32 = 0.0f0

    rast(shape, w, h, @inline((i, j, u, v) -> begin
        @inbounds total = total + absdiff(colour, target[i, j]) - absdiff(background[i, j], target[i, j])
    end))

    total
end

function averagecolor(target, shape)
    w, h = size(target)

    col::RGBA{Float32} = RGBA{Float32}(0, 0, 0, 0)
    n::Int = 0
    rast(shape, w, h, @inline((i, j, u, v) -> begin
        @inbounds col = col + target[i, j]
        n = n + 1
    end))

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
