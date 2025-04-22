#include <gtest/gtest.h>

#include "impl.h"
#include "utils.h"

TEST(CompositTest, Empty) {
    // Empty test
    int x = add(5, 4);
    EXPECT_EQ(x, mul(3, 3));
}
