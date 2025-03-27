#include "benchmark_triangle_rasterization.h"
#include "benchmark_cairo.h"
#include "benchmark_erkaman_sse-avx_rasterization.h"
#include "benchmark_rasterize.h"

int main() {
    BenchmarkRunner runner;
    runner.add_runner(std::make_shared<CairoRasterImpl>());
    runner.add_runner(std::make_shared<PolypaintRasterImpl<RasterStrategy::Bounded>>());
    runner.add_runner(std::make_shared<PolypaintRasterImpl<RasterStrategy::Integer>>());
    runner.add_runner(std::make_shared<PolypaintRasterImpl<RasterStrategy::Scanline>>());
    runner.add_runner(std::make_shared<PolypaintRasterImpl<RasterStrategy::EdgeTable>>());
    runner.add_runner(std::make_shared<PolypaintRasterImpl<RasterStrategy::Binned>>());
    runner.add_runner(std::make_shared<RasterImplPlain>());
    runner.add_runner(std::make_shared<RasterImplSSE>());
    runner.add_runner(std::make_shared<RasterImplAVX>());
    const int N = 1000;
    std::cout << "Start 64\n";
    runner.run_benchmarks(N, 0, 64);
    std::cout << "Start 128\n";
    runner.run_benchmarks(N, 0, 128);
    std::cout << "Start 256\n";
    runner.run_benchmarks(N, 0, 256);
    std::cout << "Start 512\n";
    runner.run_benchmarks(N, 0, 512);
    std::cout << "Saving outputs\n";
    runner.save_records("benchmark_results.csv", "benchmark_results.json");
    return 0;
}
