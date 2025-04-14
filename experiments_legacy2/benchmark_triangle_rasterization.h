#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>
#include <memory>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "image.h"

struct SampleInput {
    float triangle[6];
    uint8_t colour_rgba[4];
};

struct RasterImpl {
    virtual void set_canvas(ImageView<RGBA255>) = 0;
    virtual void render(SampleInput) = 0;
    virtual void teardown() {}
    virtual std::string name() const = 0;
};

std::vector<SampleInput> generate_samples(int N, int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    std::uniform_int_distribution<uint8_t> dist_colour(0, 255);

    std::vector<SampleInput> samples;
    for (int i = 0; i < N; i++) {
        SampleInput sample;
        for (int j = 0; j < 6; j++) {
            sample.triangle[j] = dist(gen);
        }
        for (int j = 0; j < 4; j++) {
            sample.colour_rgba[j] = dist_colour(gen);
        }
        samples.push_back(sample);
    }
    return samples;
}

void clear_canvas(ImageView<RGBA255> canvas) {
    std::fill(canvas.begin(), canvas.end(), RGBA255{0, 0, 0, 255});
}

struct BenchmarkRecord {
    // Metadata
    std::string runner_name;
    int canvas_width;
    int canvas_height;
    int num_trials;
    int seed;

    // Inputs
    std::vector<SampleInput> input_samples;

    // Results
    std::vector<double> times;
    double total_time;
    double iterations_per_second;
};

BenchmarkRecord benchmark_rasterization(
    std::shared_ptr<RasterImpl> raster_impl,
    ImageView<RGBA255> canvas,
    int N,
    int seed) {

    BenchmarkRecord record;
    record.runner_name = raster_impl->name();
    record.canvas_width = canvas.width();
    record.canvas_height = canvas.height();
    record.num_trials = N;
    record.seed = seed;
    
    record.input_samples = generate_samples(N, seed);

    raster_impl->set_canvas(canvas);
    for (int i = 0; i < N; i++) {
        clear_canvas(canvas);
        auto start = std::chrono::high_resolution_clock::now();
        raster_impl->render(record.input_samples[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        record.times.push_back(duration.count());
    }

    record.total_time = std::accumulate(
        record.times.begin(), record.times.end(), 0.0);
    
    record.iterations_per_second = N / record.total_time;

    raster_impl->teardown();
    return record;
}

struct BenchmarkRunner {
    std::vector<std::shared_ptr<RasterImpl>> m_runners;
    std::vector<BenchmarkRecord> m_records;

    BenchmarkRunner() {}

    void add_runner(std::shared_ptr<RasterImpl> runner) {
        m_runners.push_back(runner);
    }

    void run_benchmarks(int N, int seed, int canvas_size) {
        ImageView<RGBA255> canvas;
        canvas.m_width = canvas_size;
        canvas.m_height = canvas_size;
        canvas.m_data = (RGBA255*) aligned_alloc(64, canvas_size * canvas_size * sizeof(RGBA255));
        for (auto runner : m_runners) {
            clear_canvas(canvas);
            m_records.push_back(benchmark_rasterization(runner, canvas, N, seed));
            save_png_rgba(runner->name() + "_" + std::to_string(N) + ".png", canvas);
        }
        free(canvas.data());
    }

    void save_records(std::string output_file_csv = "", std::string output_file_json = "") const {
        if (!output_file_csv.empty()) {
            // Dump a CSV summary of the benchmark results
            std::ofstream out(output_file_csv);
            out << "runner_name,canvas_width,canvas_height,num_trials,seed,total_time,iterations_per_second\n";
            for (const auto& record : m_records) {
                out << record.runner_name << ","
                << record.canvas_width << ","
                << record.canvas_height << ","
                << record.num_trials << ","
                << record.seed << ","
                << record.total_time << ","
                << record.iterations_per_second;
                out << '\n';
            }
        }
        if (!output_file_json.empty()) {
            // Dump a JSON summary of the full benchmark results
            nlohmann::json json;

            json["records"] = nlohmann::json::array();
            for (const auto& record : m_records) {
                nlohmann::json record_json;
                record_json["runner_name"] = record.runner_name;
                record_json["canvas_width"] = record.canvas_width;
                record_json["canvas_height"] = record.canvas_height;
                record_json["num_trials"] = record.num_trials;
                record_json["seed"] = record.seed;
                record_json["total_time"] = record.total_time;
                record_json["iterations_per_second"] = record.iterations_per_second;
                record_json["times"] = record.times;
                record_json["input_samples"] = nlohmann::json::array();
                for (const auto& sample : record.input_samples) {
                    nlohmann::json sample_json;
                    sample_json["triangle"] = {sample.triangle[0], sample.triangle[1], sample.triangle[2], sample.triangle[3], sample.triangle[4], sample.triangle[5]};
                    sample_json["colour_rgba"] = {sample.colour_rgba[0], sample.colour_rgba[1], sample.colour_rgba[2], sample.colour_rgba[3]};
                    record_json["input_samples"].push_back(sample_json);
                }
                json["records"].push_back(record_json);
            }
            
            std::ofstream(output_file_json) << json.dump(4) << std::endl;
        }
    }
};

// void default_benchmark_main(std::shared_ptr<RasterImpl> raster_impl) {
//     ImageView<RGBA255> canvas;
//     canvas.m_width = 500;
//     canvas.m_height = 500;
//     canvas.m_data = (RGBA255*) aligned_alloc(64, canvas.width() * canvas.height() * sizeof(RGBA255));
//     benchmark_rasterization(raster_impl, canvas, "output.png", 100000, 0);
//     free(canvas.data());
// }
