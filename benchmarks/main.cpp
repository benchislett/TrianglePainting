#include <benchmark/benchmark.h>

#include <random>
#include <cstring>
#include <cstdlib>
#include <cstdint>

struct RGBfloat {
    float r, g, b;
};

struct RGBAfloat {
    float r, g, b, a;
};

RGBAfloat blend_RGBAfloat_over_RGBAfloat(RGBAfloat foreground, RGBAfloat background) {
    float alpha = foreground.a + background.a * (1 - foreground.a);
    return RGBAfloat{
            (foreground.r * foreground.a + background.r * background.a * (1 - foreground.a)) / alpha,
            (foreground.g * foreground.a + background.g * background.a * (1 - foreground.a)) / alpha,
            (foreground.b * foreground.a + background.b * background.a * (1 - foreground.a)) / alpha,
            alpha
    };
}

RGBfloat blend_RGBAfloat_over_RGBfloat(RGBAfloat foreground, RGBfloat background) {
    return RGBfloat{
            foreground.r * foreground.a + background.r * (1 - foreground.a),
            foreground.g * foreground.a + background.g * (1 - foreground.a),
            foreground.b * foreground.a + background.b * (1 - foreground.a)
    };
}

struct RGBu8 {
    uint8_t r, g, b;
};

struct RGBAu8 {
    uint8_t r, g, b, a;
};

RGBAu8 blend_RGBAu8_over_RGBAu8_floatintermediate(RGBAu8 foreground, RGBAu8 background) {
    // Normalize each channel to the range [0, 1]
    float f_r = foreground.r / 255.0f;
    float f_g = foreground.g / 255.0f;
    float f_b = foreground.b / 255.0f;
    float f_a = foreground.a / 255.0f;

    float b_r = background.r / 255.0f;
    float b_g = background.g / 255.0f;
    float b_b = background.b / 255.0f;
    float b_a = background.a / 255.0f;

    // Compute output alpha
    float out_a = f_a + b_a * (1.0f - f_a);

    float out_r = 0.0f, out_g = 0.0f, out_b = 0.0f;
    if (out_a > 0.0f) {
        // Compute each color channel using the "over" formula
        out_r = (f_r * f_a + b_r * b_a * (1.0f - f_a)) / out_a;
        out_g = (f_g * f_a + b_g * b_a * (1.0f - f_a)) / out_a;
        out_b = (f_b * f_a + b_b * b_a * (1.0f - f_a)) / out_a;
    }

    // Convert channels back to [0, 255] with rounding.
    RGBAu8 result;
    result.r = static_cast<uint8_t>(out_r * 255.0f + 0.5f);
    result.g = static_cast<uint8_t>(out_g * 255.0f + 0.5f);
    result.b = static_cast<uint8_t>(out_b * 255.0f + 0.5f);
    result.a = static_cast<uint8_t>(out_a * 255.0f + 0.5f);
    return result;
}

RGBu8 blend_RGBAu8_over_RGBu8_floatintermediate(RGBAu8 foreground, RGBu8 background) {
    // Normalize foreground channels and alpha to [0,1]
    float f_r = foreground.r / 255.0f;
    float f_g = foreground.g / 255.0f;
    float f_b = foreground.b / 255.0f;
    float f_a = foreground.a / 255.0f;

    // Normalize background color channels to [0,1]
    float b_r = background.r / 255.0f;
    float b_g = background.g / 255.0f;
    float b_b = background.b / 255.0f;

    // Since the background is opaque, A_background is 1.
    float out_r = f_r * f_a + b_r * (1.0f - f_a);
    float out_g = f_g * f_a + b_g * (1.0f - f_a);
    float out_b = f_b * f_a + b_b * (1.0f - f_a);

    // Convert back to 8-bit values
    RGBu8 result;
    result.r = static_cast<uint8_t>(out_r * 255.0f + 0.5f);
    result.g = static_cast<uint8_t>(out_g * 255.0f + 0.5f);
    result.b = static_cast<uint8_t>(out_b * 255.0f + 0.5f);
    return result;
}


float random_float(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int random_int(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

RGBAfloat random_rgbafloat() {
    return RGBAfloat{
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f)
    };
}

RGBfloat random_rgbfloat() {
    return RGBfloat{
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
        random_float(0.01f, 0.99f),
    };
}

RGBAu8 random_rgba_u8() {
    return RGBAu8{
        (unsigned char)random_int(1, 255),
        (unsigned char)random_int(1, 255),
        (unsigned char)random_int(1, 255),
        (unsigned char)random_int(1, 255)
    };
}

RGBu8 random_rgb_u8() {
    return RGBu8{
        (unsigned char)random_int(1, 255),
        (unsigned char)random_int(1, 255),
        (unsigned char)random_int(1, 255)
    };
}


template<typename T>
T random_T() {
    if constexpr (std::is_same_v<T, RGBAfloat>) {
        return random_rgbafloat();
    } else if constexpr (std::is_same_v<T, RGBfloat>) {
        return random_rgbfloat();
    } else if constexpr (std::is_same_v<T, RGBAu8>) {
        return random_rgba_u8();
    } else if constexpr (std::is_same_v<T, RGBu8>) {
        return random_rgb_u8();
    } else {
        static_assert("Unsupported type");
    }
}

template<typename T_fg, typename T_bg>
T_bg memcpy_shader(T_fg foreground, T_bg background) {
    if constexpr (std::is_same_v<T_fg, RGBAfloat> && std::is_same_v<T_bg, RGBAfloat>) {
        return foreground;
    } else if constexpr (std::is_same_v<T_fg, RGBfloat> && std::is_same_v<T_bg, RGBfloat>) {
        return T_bg{foreground.r, foreground.g, foreground.b};
    } else if constexpr (std::is_same_v<T_fg, RGBAu8> && std::is_same_v<T_bg, RGBAu8>) {
        return foreground;
    } else if constexpr (std::is_same_v<T_fg, RGBu8> && std::is_same_v<T_bg, RGBu8>) {
        return T_bg{foreground.r, foreground.g, foreground.b};
    } else {
        static_assert("Unsupported shader");
        return T_bg{};
    }
}

template<typename T_fg, typename T_bg, typename Shader>
static void BM_Composit_Fg_Bg(benchmark::State& state, Shader shader) {
    const int width = state.range(0);
    const int N = width * width;

    // assert N is a power of two
    assert((N & (N - 1)) == 0);

    T_bg* backgrounds = (T_bg*) aligned_alloc(std::min(128, N), N * sizeof(T_bg));
    T_fg* foregrounds = (T_fg*) aligned_alloc(std::min(128, N), N * sizeof(T_fg));

    for (int i = 0; i < N; ++i) {
        backgrounds[i] = random_T<T_bg>();
        foregrounds[i] = random_T<T_fg>();
    }
    
    for (auto _ : state) {
        for (int i = 0; i < N; ++i) {
            T_bg result = shader(foregrounds[i], backgrounds[i]);
            benchmark::DoNotOptimize(result);
            backgrounds[i] = result;
        }
    }

    state.SetItemsProcessed(N * uint64_t(state.iterations()));
    state.SetBytesProcessed(uint64_t(N) * uint64_t(state.iterations()) * sizeof(T_bg));

    free(backgrounds);
    free(foregrounds);
}

static void BM_Composit_RGBAfloat_over_RGBAfloat_memcpy(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAfloat, RGBAfloat>(state, memcpy_shader<RGBAfloat, RGBAfloat>);
}

static void BM_Composit_RGBAfloat_over_RGBAfloat_blend(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAfloat, RGBAfloat>(state, blend_RGBAfloat_over_RGBAfloat);
}

static void BM_Composit_RGBAfloat_over_RGBfloat_memcpy(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAfloat, RGBfloat>(state, memcpy_shader<RGBAfloat, RGBfloat>);
}

static void BM_Composit_RGBAfloat_over_RGBfloat_blend(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAfloat, RGBfloat>(state, blend_RGBAfloat_over_RGBfloat);
}

static void BM_Composit_RGBAu8_over_RGBAu8_memcpy(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAu8, RGBAu8>(state, memcpy_shader<RGBAu8, RGBAu8>);
}

static void BM_Composit_RGBAu8_over_RGBAu8_blend_floatintermediate(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAu8, RGBAu8>(state, blend_RGBAu8_over_RGBAu8_floatintermediate);
}

static void BM_Composit_RGBAu8_over_RGBu8_memcpy(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAu8, RGBu8>(state, memcpy_shader<RGBAu8, RGBu8>);
}

static void BM_Composit_RGBAu8_over_RGBu8_blend_floatintermediate(benchmark::State& state) {
    BM_Composit_Fg_Bg<RGBAu8, RGBu8>(state, blend_RGBAu8_over_RGBu8_floatintermediate);
}

BENCHMARK(BM_Composit_RGBAfloat_over_RGBAfloat_memcpy)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAfloat_over_RGBAfloat_blend)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAfloat_over_RGBfloat_memcpy)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAfloat_over_RGBfloat_blend)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAu8_over_RGBAu8_memcpy)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAu8_over_RGBAu8_blend_floatintermediate)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAu8_over_RGBu8_memcpy)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK(BM_Composit_RGBAu8_over_RGBu8_blend_floatintermediate)
    ->RangeMultiplier(2)
    ->Range(64, 512);

BENCHMARK_MAIN();
