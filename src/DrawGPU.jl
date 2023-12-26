module GPUDraw2D

using Images

using ..Shapes2D
using ..Spatial2D
using ..Pixel
using ..Raster2D
using ..Draw2D

using CUDA
using LLVM

import ..Draw2D: drawloss_batch, averagepixel_batch

export RasterAlgorithmGPU
# export drawloss_batch, averagepixel_batch

"""Indication that batch rasterization should be done on the GPU. Defaults to Pointwise rasterization if CUDA unavailable."""
struct RasterAlgorithmGPU <: RasterAlgorithm end

function warpreducesum(amt)
    mask = CUDA.active_mask()
    amt += CUDA.shfl_down_sync(mask, amt, 16)
    amt += CUDA.shfl_down_sync(mask, amt, 8)
    amt += CUDA.shfl_down_sync(mask, amt, 4)
    amt += CUDA.shfl_down_sync(mask, amt, 2)
    amt += CUDA.shfl_down_sync(mask, amt, 1)
    return amt
end

function warpbroadcast(val)
    return CUDA.shfl_sync(CUDA.active_mask(), val, 1)
end

function loop_bounds(shape, imgsize)
    w::Int32, h::Int32 = imgsize
    bbox = AABB(shape)
        
    minx = first(bbox.min)
    miny = last(bbox.min)
    maxx = first(bbox.max)
    maxy = last(bbox.max)

    xfloor::Int32 = max(Int32(1), u2x(minx, w)) - Int32(1)
    xceil::Int32 = min(w, u2x(maxx, w) + Int32(1)) - Int32(1)

    yfloor::Int32 = max(Int32(1), v2y(miny, h))
    yceil::Int32 = min(h, v2y(maxy, h) + Int32(1))

    xfloor, xceil, yfloor, yceil
end

function drawloss_gpu_kernel!(tris, cols, losses, target_d, img_d, lossstate)
    @inbounds begin
        tid::Int32 = threadIdx().x
        bid::Int32 = blockIdx().x

        shape::Triangle = tris[bid]
        colour::RGB{Float32} = cols[bid]
        
        w::Int32, h::Int32 = size(target_d)
        xfloor, xceil, yfloor, yceil = loop_bounds(shape, size(target_d))
        
        lossval::Float32 = 0.0f0

        xbase::Int32 = xfloor
        while xbase <= xceil
            x = xbase + tid
            if x <= w
                for y=yfloor:yceil
                    if covers(shape, Point(x2u(x, w), y2v(y, h)))
                        lossval += loss(colour, target_d[x, y], lossstate) - loss(img_d[x, y], target_d[x, y], lossstate)
                    end
                end
            end
            xbase += 32
        end

        lossval = warpreducesum(lossval)

        if tid == 1
            losses[bid] = lossval
        end
    end

    nothing
end

function averagepixel_gpu_kernel!(target_d, background_d, alpha::Float32, shapes_d, cols_d_out)
    @inbounds begin
        tid = threadIdx().x
        bid = blockIdx().x
        LLVM.Interop.assume(alpha != 0)

        shape::Triangle = shapes_d[bid]
        
        w::Int32, h::Int32 = size(target_d)
        xfloor, xceil, yfloor, yceil = loop_bounds(shape, size(target_d))
        
        col::RGB{Float32} = zero(RGB{Float32})
        amt::Int32 = 0

        xbase::Int32 = xfloor
        while xbase <= xceil
            x = xbase + tid
            if x <= w
                for y=yfloor:yceil
                    if covers(shape, Point(x2u(x, w), y2v(y, h)))
                        col += ((target_d[x, y] - ((1.0f0 - alpha) .* background_d[x, y])) ./ alpha)
                        amt += Int32(1)
                    end
                end
            end
            xbase += 32
        end

        col = RGB{Float32}(warpreducesum(col.r), warpreducesum(col.g), warpreducesum(col.b))
        amt = warpreducesum(amt)
        denom::Float32 = Float32(max(Int32(1), amt))

        LLVM.Interop.assume(denom != 0)
        col = col ./ denom

        if tid == Int32(1)
            cols_d_out[bid] = col
        end

    end
    nothing
end

function drawloss_gpu!(tris_d, cols_d, losses_d_out, target_d, img_d, lossstate)
    CUDA.@sync begin @cuda threads=32 blocks=length(tris_d) always_inline=true drawloss_gpu_kernel!(tris_d, cols_d, losses_d_out, target_d, img_d, lossstate) end
end

function averagepixel_gpu!(target_d, background_d, alpha, shapes_d, cols_d_out)
    CUDA.@sync begin @cuda threads=32 blocks=length(shapes_d) always_inline=true averagepixel_gpu_kernel!(target_d, background_d, alpha, shapes_d, cols_d_out) end
end

function drawloss_batch(target, background, shapes, colours, lossstate, ::RasterAlgorithmGPU)
    if !CUDA.functional()
        return drawloss_batch(target, background, shapes, colours, lossstate, RasterAlgorithmPointwise())
    end

    losses_d = CuArray{Float32}(undef, length(shapes))
    drawloss_gpu!(cu(shapes), cu(colours), losses_d, cu(target), cu(background), lossstate)
    Array(losses_d)
end

function averagepixel_batch(target, background, alpha, shapes, ::RasterAlgorithmGPU)
    if !CUDA.functional()
        return averagepixel_batch(target, background, alpha, shapes, RasterAlgorithmPointwise())
    end

    pixels_d = CuArray{RGB{Float32}}(undef, length(shapes))
    averagepixel_gpu!(cu(target), cu(background), alpha, cu(shapes), pixels_d)
    collect(map(pix -> RGBA{Float32}(pix.r, pix.g, pix.b, alpha), Array(pixels_d)))
end

end