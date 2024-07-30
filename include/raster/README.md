# Rasterization

## Parameters

The parameters of a rasterizer:

- Polygon shape to be rasterized (defaults to Triangle)
- Primitive batch support (defaults to supporting multiple polygons of the same shape)
- Transparent primitive support (defaults to RGBA)
- Transparent output buffer support (defaults to RGBA)
- Dimension (defaults to 2D)
- Implementation strategy (GPU/CPU, choice of raster algorithm)

## Benchmarks

| Name | 1 Triangle @ 128x128x4 | 1 Triangle @ 1024x1024x4 | 100 Triangles @ 128x128x4 | 100 Triangles @ 1024x1024x4 |
| ---- | ---------------------- | ------------------------ | ------------------------- | --------------------------- |
|      |                        |                          |                           |                             |