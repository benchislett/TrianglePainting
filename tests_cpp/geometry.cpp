#include "geometry.h"

#include <gtest/gtest.h>

#define BARY_TEST_EPSILON 1e-6

// Compute barycentric coordinates and verify that the interpolation formula holds
Barycentric bary_test(const Triangle& tri, const Point& p) {
  auto bary = barycentric_coordinates(p, tri);
  float interp_x = bary.w * tri[0].x + bary.u * tri[1].x + bary.v * tri[2].x;
  float interp_y = bary.w * tri[0].y + bary.u * tri[1].y + bary.v * tri[2].y;
  EXPECT_NEAR(interp_x, p.x, BARY_TEST_EPSILON);
  EXPECT_NEAR(interp_y, p.y, BARY_TEST_EPSILON);
  return bary;
}

void bary_test_eq(const Triangle& tri, const Point& p, const Barycentric& expected) {
  auto bary = bary_test(tri, p);
  EXPECT_NEAR(bary.u, expected.u, BARY_TEST_EPSILON);
  EXPECT_NEAR(bary.v, expected.v, BARY_TEST_EPSILON);
  EXPECT_NEAR(bary.w, expected.w, BARY_TEST_EPSILON);
}

TEST(Geometry, BarycentricTest) {
  // manual tests ensuring consistency with reference values
  Triangle tri;
  tri.vertices = std::array<Point, 3>{Point{0, 0}, Point{1, 0}, Point{0, 1}};

  bary_test_eq(tri, {0, 0}, {0, 0, 1});
  bary_test_eq(tri, {1, 0}, {1, 0, 0});
  bary_test_eq(tri, {0, 1}, {0, 1, 0});

  // automated grid test to ensure consistency with interpolation formula per the docs
  tri.vertices = std::array<Point, 3>{Point{-1, -1}, Point{1, -1}, Point{-1, 1}};
  for (float x = -3; x < 3; x += 0.1) {
    for (float y = -3; y < 3; y += 0.1) {
      bary_test(tri, {x, y});
    }
  }
}

TEST(Geometry, PointInCircle) {
    Circle c;
    c.center = Point{0,0};
    c.radius = 2.0;
    EXPECT_TRUE(c.is_inside({0,0}));
    EXPECT_TRUE(c.is_inside({1,1}));
    EXPECT_FALSE(c.is_inside({3,0}));
    EXPECT_TRUE(c.is_inside({1.99f, 0}));
    EXPECT_TRUE(c.is_inside({-1.5f, 1.0f}));
    EXPECT_FALSE(c.is_inside({2.1f, 2.1f}));
    EXPECT_FALSE(c.is_inside({-3, -1}));
}

TEST(Geometry, PointInTriangle) {
    Triangle tri;
    tri.vertices = std::array<Point,3>{Point{0,0}, Point{4,0}, Point{0,4}};
    EXPECT_TRUE(tri.is_inside({1,1}));
    EXPECT_FALSE(tri.is_inside({5,5}));
    EXPECT_TRUE(tri.is_inside({2,0.1f}));
    EXPECT_TRUE(tri.is_inside({0.1f,3}));
    EXPECT_FALSE(tri.is_inside({4,1}));
    EXPECT_FALSE(tri.is_inside({-1,-1}));
}

TEST(Geometry, PointInPolygon) {
    Polygon poly;
    poly.vertices = {{0,0},{4,0},{4,4},{0,4}};
    EXPECT_TRUE(poly.is_inside({2,2}));
    EXPECT_FALSE(poly.is_inside({-1,2}));
    EXPECT_TRUE(poly.is_inside({1,1}));
    EXPECT_TRUE(poly.is_inside({3.5f, 0.5f}));
    EXPECT_FALSE(poly.is_inside({4.1f,4.1f}));
    EXPECT_FALSE(poly.is_inside({-2,2}));
}
