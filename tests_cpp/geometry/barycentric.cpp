#include "geometry/barycentric.h"

#include <gtest/gtest.h>

#define BARY_TEST_EPSILON 1e-6

// Compute barycentric coordinates and verify that the interpolation formula holds
geometry::barycentric bary_test(const geometry::triangle& tri, const geometry::point& p) {
  auto bary = geometry::barycentric_coordinates(p, tri);
  float interp_x = bary.w * tri.a.x + bary.u * tri.b.x + bary.v * tri.c.x;
  float interp_y = bary.w * tri.a.y + bary.u * tri.b.y + bary.v * tri.c.y;
  EXPECT_NEAR(interp_x, p.x, BARY_TEST_EPSILON);
  EXPECT_NEAR(interp_y, p.y, BARY_TEST_EPSILON);
  return bary;
}

void bary_test_eq(const geometry::triangle& tri, const geometry::point& p, const geometry::barycentric& expected) {
  auto bary = bary_test(tri, p);
  EXPECT_NEAR(bary.u, expected.u, BARY_TEST_EPSILON);
  EXPECT_NEAR(bary.v, expected.v, BARY_TEST_EPSILON);
  EXPECT_NEAR(bary.w, expected.w, BARY_TEST_EPSILON);
}

TEST(Geometry, BarycentricTest) {
  // manual tests ensuring consistency with reference values
  geometry::triangle tri{{0, 0}, {1, 0}, {0, 1}};

  bary_test_eq(tri, {0, 0}, {0, 0, 1});
  bary_test_eq(tri, {1, 0}, {1, 0, 0});
  bary_test_eq(tri, {0, 1}, {0, 1, 0});

  // automated grid test to ensure consistency with interpolation formula per the docs
  tri = {{-1, -1}, {1, -1}, {-1, 1}};
  for (float x = -3; x < 3; x += 0.1) {
    for (float y = -3; y < 3; y += 0.1) {
      bary_test(tri, {x, y});
    }
  }
}
