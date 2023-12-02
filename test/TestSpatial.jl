import Test: @test, @testset

using Paint.Shapes2D
using Paint.Spatial2D

@testset verbose = true "Right Triangle Containment" begin
    tri = Triangle{Float64}([Pair(0.0, 0.0), Pair(1.0, 0.0), Pair(0.0, 1.0)])

    @testset "Contains points near origin" begin
        @test Spatial2D.contains(tri, Pair(0.00001, 0.00001))
        @test Spatial2D.contains(tri, Pair(0.1, 0.1))
    end

    @testset "Does not contain points past origin" begin
        @test !Spatial2D.contains(tri, Pair(-0.1, 0.1))
        @test !Spatial2D.contains(tri, Pair(0.1, -0.1))
        @test !Spatial2D.contains(tri, Pair(-0.1, -0.1))
        @test !Spatial2D.contains(tri, Pair(-0.00001, -0.00001))
    end

    @testset "Contains only points below hypotenuse" begin
        @test !Spatial2D.contains(tri, Pair(0.500001, 0.500001))
        @test Spatial2D.contains(tri, Pair(0.49999, 0.49999))
    end
end

nothing