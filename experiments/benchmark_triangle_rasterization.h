#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <memory>
#include <string>
#include "../include/image.h"

struct SampleInput {
    float triangle[6];
    uint8_t colour_rgba[4];
};

struct RasterImpl {
    virtual void set_canvas(ImageView<RGBA255>) = 0;
    virtual void render(SampleInput) = 0;
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

struct BenchmarkOutput {
    std::vector<double> times;
    double total_time;
    double iterations_per_second;
};

void benchmark_rasterization(
    std::shared_ptr<RasterImpl> raster_impl,
    ImageView<RGBA255> canvas,
    std::string output_filename,
    int N,
    int seed = 0) {
    auto samples = generate_samples(N, seed);

    BenchmarkOutput output;
    raster_impl->set_canvas(canvas);
    // auto true_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        clear_canvas(canvas);
        auto start = std::chrono::high_resolution_clock::now();
        raster_impl->render(samples[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        output.times.push_back(duration.count());
    }
    output.total_time = std::accumulate(output.times.begin(), output.times.end(), 0.0);
    output.iterations_per_second = N / output.total_time;

    // auto true_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> true_duration = true_end - true_start;
    // std::cout << "Total time: " << true_duration.count() << "s" << std::endl;
    // std::cout << "Iterations/s: " << N / true_duration.count() << std::endl;

    // Pretty-print the output
    std::cout << std::setw(15) << "Iteration"
              << std::setw(15) << "Time (s)" << std::endl;
    
    // std::cout << std::string(30, '-') << std::endl;
    // for (size_t i = 0; i < output.times.size(); ++i) {
    //     std::cout << std::setw(15) << i + 1
    //               << std::setw(15) << std::fixed << std::setprecision(6) << output.times[i]
    //               << std::endl;
    // }

    std::cout << std::string(30, '-') << std::endl;
    std::cout << std::setw(15) << "Total Time"
              << std::setw(15) << std::fixed << std::setprecision(6) << output.total_time
              << std::endl;
    std::cout << std::setw(15) << "Iter/s"
              << std::setw(15) << std::fixed << std::setprecision(6) << output.iterations_per_second
              << std::endl;
    
    // Save the output to a file
    save_png_rgba(output_filename.c_str(), canvas);
}

void default_benchmark_main(std::shared_ptr<RasterImpl> raster_impl) {
    Image<RGBA255> canvas(500, 500);
    benchmark_rasterization(raster_impl, canvas, "output.png", 100000, 0);
}
