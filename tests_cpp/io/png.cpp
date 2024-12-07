#include "io/png.h"

#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cstdio>

const std::string data_prefix = "tests_cpp/data/";
const std::string temp_prefix = "tests_cpp/temp/";

std::string project_base_path() {
    std::string current_path = std::filesystem::current_path();
    std::string project_base_path = current_path.substr(0, current_path.find("build"));
    if (project_base_path.back() != '/') {
        project_base_path += "/";
    }
    return project_base_path;
}

TEST(IO, PNG) {
    std::vector<std::string> test_cases = {"basn0g08", "basn2c08", "basn4a08", "basn6a08"};

    for (std::string& name : test_cases) {
        std::string json_in_path = project_base_path() + data_prefix + name + ".json";
        std::string png_in_path = project_base_path() + data_prefix + name + ".png";
        std::string png_temp_path = project_base_path() + temp_prefix + name + ".png";

        nlohmann::json ref_json;
        try {
            std::ifstream f(json_in_path);
            ref_json = nlohmann::json::parse(f);
        } catch (const std::exception& e) {
            std::cerr << "Failed to open test file. Make sure that the tests are run from the project root or build directory. Error message:\n";
            std::cerr << e.what() << std::endl;
            throw e;
        }

        io::Image<io::RGBA255> img =io::load_png_rgba(png_in_path);;
        for (int i = 0; i < img.size(); i++) {
            EXPECT_EQ(img[i].r, ref_json[4*i+0]);
            EXPECT_EQ(img[i].g, ref_json[4*i+1]);
            EXPECT_EQ(img[i].b, ref_json[4*i+2]);
            EXPECT_EQ(img[i].a, ref_json[4*i+3]);
        }

        io::save_png_rgba(png_temp_path, img);
        io::Image<io::RGBA255> new_img = io::load_png_rgba(png_temp_path);

        for (int i = 0; i < new_img.size(); i++) {
            EXPECT_EQ(new_img[i].r, ref_json[4*i+0]);
            EXPECT_EQ(new_img[i].g, ref_json[4*i+1]);
            EXPECT_EQ(new_img[i].b, ref_json[4*i+2]);
            EXPECT_EQ(new_img[i].a, ref_json[4*i+3]);
        }

        std::remove(png_temp_path.c_str());
    }
}
