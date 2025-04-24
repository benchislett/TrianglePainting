#include <gtest/gtest.h>
#include "impl.h"
#include "utils.h"
#include "lodepng.h"

TEST(PremultipliedBlendDrawTest, DrawFillRGBAOverIMRGB) {
    int resolution = 512;

    unsigned char* buffer_rgb_ref = (unsigned char*)allocate_aligned(resolution * resolution * 3);
    unsigned char* buffer_rgb_avx512 = (unsigned char*)allocate_aligned(resolution * resolution * 3);
    
    for (int i = 0; i < resolution * resolution; ++i) {
        unsigned char value_r = random_u8();
        unsigned char value_g = random_u8();
        unsigned char value_b = random_u8();
        buffer_rgb_ref[i * 3 + 0] = value_r;
        buffer_rgb_ref[i * 3 + 1] = value_g;
        buffer_rgb_ref[i * 3 + 2] = value_b;

        int block_id = i / 32;
        int block_pos = i % 32;
        buffer_rgb_avx512[block_id * 32 * 3 +  0 + block_pos] = value_r;
        buffer_rgb_avx512[block_id * 32 * 3 + 32 + block_pos] = value_g;
        buffer_rgb_avx512[block_id * 32 * 3 + 64 + block_pos] = value_b;
    }

    unsigned char foreground[4] = { 82, 61, 0, 120 };
    draw_fill_rgba_over_imrgb_scalar(buffer_rgb_ref, foreground, 512);

    draw_fill_rgba_over_imrgb_avx512(buffer_rgb_avx512, foreground, 512);
    for (int i = 0; i < resolution * resolution; ++i) {
        unsigned char ref_r = buffer_rgb_ref[i * 3];
        unsigned char ref_g = buffer_rgb_ref[i * 3 + 1];
        unsigned char ref_b = buffer_rgb_ref[i * 3 + 2];

        // account for modified indexing of avx512
        int block_id = i / 32;
        int block_pos = i % 32;
        unsigned char avx512_r = buffer_rgb_avx512[block_id * 32 * 3 +  0 + block_pos];
        unsigned char avx512_g = buffer_rgb_avx512[block_id * 32 * 3 + 32 + block_pos];
        unsigned char avx512_b = buffer_rgb_avx512[block_id * 32 * 3 + 64 + block_pos];

        ASSERT_EQ(ref_r, avx512_r) << "Mismatch at index " << i << " for red channel";
        ASSERT_EQ(ref_g, avx512_g) << "Mismatch at index " << i << " for green channel";
        ASSERT_EQ(ref_b, avx512_b) << "Mismatch at index " << i << " for blue channel";
    }

    // write buffer_rgb_ref to a png file using lodepng
    unsigned char* image = (unsigned char*)malloc(resolution * resolution * 4);
    for (int i = 0; i < resolution * resolution; ++i) {
        image[i * 4 + 0] = buffer_rgb_ref[i * 3 + 0];
        image[i * 4 + 1] = buffer_rgb_ref[i * 3 + 1];
        image[i * 4 + 2] = buffer_rgb_ref[i * 3 + 2];
        image[i * 4 + 3] = 255;
    }
    lodepng_encode32_file("output_drawfill_ref.png", image, resolution, resolution);
    free(image);
    free(buffer_rgb_ref);
    free(buffer_rgb_avx512);
}

TEST(PremultipliedBlendDrawTest, DrawTriangleRGBAOverIMRGB) {
    int resolution = 512;

    unsigned char* buffer_rgb_ref = (unsigned char*)allocate_aligned(resolution * resolution * 3);
    unsigned char* buffer_rgb_avx512 = (unsigned char*)allocate_aligned(resolution * resolution * 3);
    
    for (int i = 0; i < resolution * resolution; ++i) {
        unsigned char value_r = random_u8();
        unsigned char value_g = random_u8();
        unsigned char value_b = random_u8();
        buffer_rgb_ref[i * 3 + 0] = value_r;
        buffer_rgb_ref[i * 3 + 1] = value_g;
        buffer_rgb_ref[i * 3 + 2] = value_b;

        int block_id = i / 32;
        int block_pos = i % 32;
        buffer_rgb_avx512[block_id * 32 * 3 +  0 + block_pos] = value_r;
        buffer_rgb_avx512[block_id * 32 * 3 + 32 + block_pos] = value_g;
        buffer_rgb_avx512[block_id * 32 * 3 + 64 + block_pos] = value_b;
    }

    unsigned char foreground[4] = { 150, 100, 60, 200 };
    draw_triangle_rgba_over_imrgb_scalar(buffer_rgb_ref, foreground, 512);
    draw_triangle_rgba_over_imrgb_avx512(buffer_rgb_avx512, foreground, 512);

    for (int i = 0; i < resolution * resolution; ++i) {
        unsigned char ref_r = buffer_rgb_ref[i * 3];
        unsigned char ref_g = buffer_rgb_ref[i * 3 + 1];
        unsigned char ref_b = buffer_rgb_ref[i * 3 + 2];

        // account for modified indexing of avx512
        int block_id = i / 32;
        int block_pos = i % 32;
        unsigned char avx512_r = buffer_rgb_avx512[block_id * 32 * 3 +  0 + block_pos];
        unsigned char avx512_g = buffer_rgb_avx512[block_id * 32 * 3 + 32 + block_pos];
        unsigned char avx512_b = buffer_rgb_avx512[block_id * 32 * 3 + 64 + block_pos];

        ASSERT_EQ(ref_r, avx512_r) << "Mismatch at index " << i << " for red channel";
        ASSERT_EQ(ref_g, avx512_g) << "Mismatch at index " << i << " for green channel";
        ASSERT_EQ(ref_b, avx512_b) << "Mismatch at index " << i << " for blue channel";
    }

    // write buffer_rgb_ref to a png file using lodepng
    unsigned char* image = (unsigned char*)malloc(resolution * resolution * 4);
    for (int i = 0; i < resolution * resolution; ++i) {
        image[i * 4 + 0] = buffer_rgb_ref[i * 3 + 0];
        image[i * 4 + 1] = buffer_rgb_ref[i * 3 + 1];
        image[i * 4 + 2] = buffer_rgb_ref[i * 3 + 2];
        image[i * 4 + 3] = 255;
    }
    lodepng_encode32_file("output_drawtriangle_ref.png", image, resolution, resolution);
    // lodepng_encode32_file("output_drawtriangle_avx512.png", buffer_rgb_avx512, resolution, resolution);
    free(image);
    free(buffer_rgb_ref);
    free(buffer_rgb_avx512);
}
