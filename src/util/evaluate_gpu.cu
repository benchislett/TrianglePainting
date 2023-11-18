#include "evaluate_cuda.h"

#include "dna.h"
#include "misc_math.h"

#include <cuda.h>
#include <cuda/std/array>
#include <cuda/std/tuple>

#include <cub/cub.cuh>

struct CuDNATri50 {
    struct Primitive {
        cuda::std::array<cuda::std::pair<float, float>, 3> vertices;
        float r, g, b;
    };

    cuda::std::array<Primitive, 50> polys;
};

__global__ void render_image(CuDNATri50 dna, float* image_out) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int i = idx / resolution;
    int j = idx % resolution;

    float u = (float) i / (float) resolution;
    float v = (float) j / (float) resolution;

    *((float3*)(&image_out[3 * idx])) = color_pixel_blend(u, v, dna);
}


__global__ void subtract_images(CuDNATri50 dna, const float *target_image, float* error) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int i = idx / resolution;
    int j = idx % resolution;

    float u = (float) i / (float) resolution;
    float v = (float) j / (float) resolution;

    float3 rgb = color_pixel_blend(u, v, dna);
    float3 target_rgb = make_float3(target_image[3 * idx], target_image[3 * idx + 1], target_image[3 * idx + 2]);

    error[idx] = abs_error(rgb, target_rgb);
}

constexpr int target_image_size = resolution * resolution * 3 * sizeof(float);

void LossState::init() {
    cudaMalloc(&target_image, target_image_size);
    cudaMalloc(&error_values, target_image_size / 3);
}

float tri_loss_gpu(const DNATri50& dna, const float *target_image, LossState& state) {
    CuDNATri50 dna_alt = *(CuDNATri50*)(&dna);

    cudaMemcpy(state.target_image, target_image, target_image_size, ::cudaMemcpyHostToDevice);

    subtract_images<<<resolution, resolution>>>(dna_alt, state.target_image, state.error_values);

    float* device_answer = nullptr;
    float* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaMalloc(&device_answer, sizeof(float));
    cudaMemset(device_answer, 0, sizeof(float));

    d_temp_storage = NULL;
    temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, state.error_values, device_answer, resolution * resolution);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, state.error_values, device_answer, resolution * resolution);

    float answer;
    cudaMemcpy(&answer, device_answer, sizeof(float), ::cudaMemcpyDeviceToHost);

    cudaFree(device_answer);
    cudaFree(d_temp_storage);

    cudaDeviceSynchronize();

    return answer;
}

float* tri_render_gpu(const DNATri50& dna) {
    CuDNATri50 dna_alt = *(CuDNATri50*)(&dna);

    constexpr int image_out_size = resolution * resolution * 3 * sizeof(float);
    float* image_out = (float*) malloc (image_out_size);

    float* device_image_out;
    cudaMalloc(&device_image_out, image_out_size);

    render_image<<<resolution, resolution>>>(dna_alt, device_image_out);

    cudaMemcpy(image_out, device_image_out, image_out_size, ::cudaMemcpyDeviceToHost);

    cudaFree(device_image_out);

    return image_out;
}
