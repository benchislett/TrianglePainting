import Test: @test, @testset

using Paint

@testset verbose = true "Right Triangle Containment" begin
    tri = Triangle([Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)])

    @testset "Contains points near origin" begin
        @test covers(tri, Point(0.00001, 0.00001))
        @test covers(tri, Point(0.1, 0.1))
    end

    @testset "Does not contain points past origin" begin
        @test !covers(tri, Point(-0.1, 0.1))
        @test !covers(tri, Point(0.1, -0.1))
        @test !covers(tri, Point(-0.1, -0.1))
        @test !covers(tri, Point(-0.00001, -0.00001))
    end

    @testset "Contains only points below hypotenuse" begin
        @test !covers(tri, Point(0.500001, 0.500001))
        @test covers(tri, Point(0.49999, 0.49999))
    end
end

nothing