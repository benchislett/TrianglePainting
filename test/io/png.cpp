#include "io/image.h"

#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include <fstream>
#include <cstdio>

const std::string data_prefix = "../test/data/";
const std::string temp_prefix = "../test/temp/";

TEST(IO, PNG) {
    std::vector<std::string> test_cases = {"basn0g08", "basn2c08", "basn4a08", "basn6a08"};

    for (std::string& name : test_cases) {
        std::ifstream f(data_prefix + name + ".json");
        auto ref_json = nlohmann::json::parse(f);
        io::Image img = io::load_png(data_prefix + name + ".png");
        EXPECT_EQ(img.channels, 4);
        for (int i = 0; i < img.width * img.height * img.channels; i++) {
            EXPECT_EQ(img.data[i], ref_json[i]);
        }

        io::save_png(temp_prefix + name + ".png", img);
        io::Image new_img = io::load_png(temp_prefix + name + ".png");

        EXPECT_EQ(new_img.channels, 4);
        for (int i = 0; i < new_img.width * new_img.height * new_img.channels; i++) {
            EXPECT_EQ(new_img.data[i], ref_json[i]);
        }

        std::remove((temp_prefix + name + ".png").c_str());
    }
}
