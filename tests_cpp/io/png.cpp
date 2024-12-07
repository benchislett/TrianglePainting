#include "io/png.h"

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
        io::Image<io::RGBA255> img = io::load_png_rgba(data_prefix + name + ".png");
        for (int i = 0; i < img.size(); i++) {
            EXPECT_EQ(img[i].r, ref_json[4*i+0]);
            EXPECT_EQ(img[i].g, ref_json[4*i+1]);
            EXPECT_EQ(img[i].b, ref_json[4*i+2]);
            EXPECT_EQ(img[i].a, ref_json[4*i+3]);
        }

        io::save_png_rgba(temp_prefix + name + ".png", img);
        io::Image<io::RGBA255> new_img = io::load_png_rgba(temp_prefix + name + ".png");

        for (int i = 0; i < new_img.size(); i++) {
            EXPECT_EQ(new_img[i].r, ref_json[4*i+0]);
            EXPECT_EQ(new_img[i].g, ref_json[4*i+1]);
            EXPECT_EQ(new_img[i].b, ref_json[4*i+2]);
            EXPECT_EQ(new_img[i].a, ref_json[4*i+3]);
        }

        std::remove((temp_prefix + name + ".png").c_str());
    }
}
