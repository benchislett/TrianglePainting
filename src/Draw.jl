module Draw2D

using Images

export draw!
export imloss, drawloss, averagecolor
export rast
export RastAlgorithmPointwise, RastAlgorithmScanline
export uv, absdiff

using ..Spatial2D
using ..Shapes2D

sqr(x) = x*x
absdiff(r1, r2) = sqr(r1.r - r2.r) + sqr(r1.g - r2.g) + sqr(r1.b - r2.b)

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

struct DrawRasterState{T<:AbstractFloat, Arr}
    img::Arr
    col::RGB{T}
end

struct DrawlossRasterState{T<:AbstractFloat, Arr}
    target::Arr
    background::Arr
    colour::RGB{T}
    total::T
end

struct ColourRasterState{T<:AbstractFloat, Arr}
    target::Arr
    colour::RGB{T}
    count::Int
end

function rasterfunc(i, j, u, v, state::DrawRasterState{T, Arr}) where {T, Arr}
    @inbounds state.img[i, j] = state.col
    state
end

function rasterfunc(i, j, u, v, state::DrawlossRasterState{T, Arr}) where {T, Arr}
    newtotal::T = state.total + absdiff(state.colour, state.target[i, j]) - absdiff(state.background[i, j], state.target[i, j])
    DrawlossRasterState{T, Arr}(state.target, state.background, state.colour, newtotal)
end

function rasterfunc(i, j, u, v, state::ColourRasterState{T, Arr}) where {T, Arr}
    @inbounds ColourRasterState{T, Arr}(state.target, state.colour + state.target[i, j], state.count + 1)
end

# http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
function rasttop(shape::Triangle{Float32}, w, h, state)
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
            state = rasterfunc(x, y, u, v, state)
        end
        y = y - 1
        curx1 -= invslope1 / Float32(h)
        curx2 -= invslope2 / Float32(h)
    end

    state
end

function rastbot(shape::Triangle{Float32}, w, h, state)
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
            state = rasterfunc(x, y, u, v, state)
        end
        y = y + 1
        curx1 += invslope1 / Float32(h)
        curx2 += invslope2 / Float32(h)
    end

    state
end

abstract type RastAlgorithm end
struct RastAlgorithmScanline <: RastAlgorithm end
struct RastAlgorithmPointwise <: RastAlgorithm end

function rast(shape, w, h, state, ::RastAlgorithmPointwise)
    minx, miny = Shapes2D.min(shape)
    maxx, maxy = Shapes2D.max(shape)

    for y in max(1, toycoord(miny, h)):min(h, toycoord(maxy, h) + 1)
        for x in max(1, toxcoord(minx, w)):min(w, toxcoord(maxx, w) + 1)
            u, v = uv(eltype(shape), x, y, w, h)
            if Spatial2D.contains(shape, Pair(u, v))
                state = rasterfunc(x, y, u, v, state)
            end
        end
    end
    state
end

function rast(shape::Triangle{Float32}, w, h, state, ::RastAlgorithmScanline)
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
        state = rastbot(Triangle{Float32}(v1, v2, v3), w, h, state)
    elseif v1.second == v2.second
        state = rasttop(Triangle{Float32}(v1, v2, v3), w, h, state)
    else
        x4 = v1.first + ((v2.second - v1.second) / (v3.second - v1.second)) * (v3.first - v1.first)
        v4 = Pair(x4, v2.second)
        state = rastbot(Triangle{Float32}(v1, v2, v4), w, h, state)
        state = rasttop(Triangle{Float32}(v2, v4, v3), w, h, state)
    end
    state
end

function draw!(img, shape, colour, alg = RastAlgorithmPointwise())
    w, h = size(img)

    state = DrawRasterState{eltype(shape), typeof(img)}(img, colour)
    rast(shape, w, h, state, alg)
end

function drawloss(target, background, shape, colour, alg = RastAlgorithmPointwise())
    w, h = size(target)

    state = DrawlossRasterState{eltype(shape), typeof(target)}(target, background, colour, zero(eltype(shape)))
    state = rast(shape, w, h, state, alg)
    state.total
end

function averagecolor(target, shape, alg = RastAlgorithmPointwise())
    w, h = size(target)

    state = ColourRasterState{eltype(shape), typeof(target)}(target, RGB{eltype(shape)}(0, 0, 0), 0)
    state = rast(shape, w, h, state, alg)
    state.colour / state.count
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
