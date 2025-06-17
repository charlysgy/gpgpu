#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <iomanip>
#include <chrono>


#include "filter_impl.h"
#include "logo.h"


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

const uint8_t LOW_THRESHOLD = 30;
const uint8_t HIGH_THRESHOLD = 50;
const float LAB_DISTANCE_THRESHOLD = 50.0f;


struct rgb
{
    uint8_t r, g, b;
};

struct PixelState
{
    float bg_L, bg_a, bg_b;
    float cand_L, cand_a, cand_b;
    int t;
    int initialized;
};

struct Point
{
    int x, y;
};

static uint8_t *d_buffer = nullptr;
static uint8_t *d_original_buffer = nullptr;
static uint8_t *d_temp_buffer1 = nullptr;
static uint8_t *d_temp_buffer2 = nullptr;
static uint8_t *d_mask_buffer = nullptr;

static PixelState *d_states = nullptr;
static Point *d_queue = nullptr;
static Point *d_next_queue = nullptr;
static int *d_queue_size = nullptr;
static int *d_next_queue_size = nullptr;

static size_t current_buffer_size = 0;
static size_t current_image_size = 0;

static float total_gpu_time = 0.0f;
static int frame_count = 0;

template <typename T>
void check(T err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        std::exit(EXIT_FAILURE);
    }
}

void initialize_gpu_memory(int width, int height, int plane_stride)
{
    size_t buffer_size = height * plane_stride;
    size_t image_size = width * height;
    
    if (current_buffer_size >= buffer_size && current_image_size >= image_size)
        return;
    
    if (d_buffer) {
        cudaFree(d_buffer);
        cudaFree(d_original_buffer);
        cudaFree(d_temp_buffer1);
        cudaFree(d_temp_buffer2);
        cudaFree(d_mask_buffer);
        cudaFree(d_states);
        cudaFree(d_queue);
        cudaFree(d_next_queue);
        cudaFree(d_queue_size);
        cudaFree(d_next_queue_size);
    }
    
    buffer_size = buffer_size * 1.2;
    image_size = image_size * 1.2;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_buffer, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_original_buffer, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_buffer1, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_buffer2, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mask_buffer, buffer_size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, image_size * sizeof(PixelState)));
    CHECK_CUDA_ERROR(cudaMemset(d_states, 0, image_size * sizeof(PixelState)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_queue, image_size * sizeof(Point)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_next_queue, image_size * sizeof(Point)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_queue_size, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_next_queue_size, sizeof(int)));
    
    current_buffer_size = buffer_size;
    current_image_size = image_size;
    
    std::cerr << "[CUDA] GPU Memory initialized: " << buffer_size / (1024*1024) << " MB\n";
}

__device__ __forceinline__ float srgb_to_linear(float c)
{
    return (c <= 0.04045f) ? __fdiv_rn(c, 12.92f)
                           : __powf(__fdiv_rn(c + 0.055f, 1.055f), 2.4f);
}

__device__ __forceinline__ void rgb_to_lab(uint8_t R, uint8_t G, uint8_t B, 
                                          float &L, float &a, float &b)
{
    const float inv255 = 0.00392156862745f; // 1.0f / 255.0f
    
    float r = srgb_to_linear(R * inv255);
    float g = srgb_to_linear(G * inv255);
    float b_ = srgb_to_linear(B * inv255);

    float X = __fmaf_rn(r, 0.4124564f, __fmaf_rn(g, 0.3575761f, b_ * 0.1804375f));
    float Y = __fmaf_rn(r, 0.2126729f, __fmaf_rn(g, 0.7151522f, b_ * 0.0721750f));
    float Z = __fmaf_rn(r, 0.0193339f, __fmaf_rn(g, 0.1191920f, b_ * 0.9503041f));

    const float inv_Xn = 1.05263157895f; // 1.0f / 0.95047f
    const float inv_Zn = 0.91841491841f; // 1.0f / 1.08883f
    
    float fx = (X * inv_Xn > 0.008856f) ? cbrtf(X * inv_Xn) 
               : __fmaf_rn(7.787f, X * inv_Xn, 0.137931034f);
    float fy = (Y > 0.008856f) ? cbrtf(Y) 
               : __fmaf_rn(7.787f, Y, 0.137931034f);
    float fz = (Z * inv_Zn > 0.008856f) ? cbrtf(Z * inv_Zn) 
               : __fmaf_rn(7.787f, Z * inv_Zn, 0.137931034f);

    L = __fmaf_rn(116.0f, fy, -16.0f);
    a = __fmul_rn(500.0f, (fx - fy));
    b = __fmul_rn(200.0f, (fy - fz));
}


__global__ void background_estimation_kernel_optimized(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ output,
    PixelState *__restrict__ states,
    int width, int height, int stride, int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * stride + x * pixel_stride;
    int state_idx = y * width + x;

    uint8_t R = input[pixel_idx];
    uint8_t G = input[pixel_idx + 1];
    uint8_t B = input[pixel_idx + 2];

    float L, a, b;
    rgb_to_lab(R, G, B, L, a, b);

    PixelState &s = states[state_idx];

    if (!s.initialized) {
        s.bg_L = L; s.bg_a = a; s.bg_b = b;
        s.t = 0;
        s.initialized = 1;
        output[pixel_idx] = 0;
    } else {
        float dL = s.bg_L - L;
        float da = s.bg_a - a;
        float db = s.bg_b - b;
        float dist = __fsqrt_rn(__fmaf_rn(dL, dL, __fmaf_rn(da, da, db * db)));
        
        bool match = dist < LAB_DISTANCE_THRESHOLD;

        if (!match) {
            if (s.t == 0) {
                s.cand_L = L; s.cand_a = a; s.cand_b = b;
                s.t = 1;
            } else if (s.t < 25) {
                s.cand_L = __fmaf_rn(s.cand_L, 0.5f, L * 0.5f);
                s.cand_a = __fmaf_rn(s.cand_a, 0.5f, a * 0.5f);
                s.cand_b = __fmaf_rn(s.cand_b, 0.5f, b * 0.5f);
                s.t++;
            } else {
                s.bg_L = s.cand_L; s.bg_a = s.cand_a; s.bg_b = s.cand_b;
                s.t = 0;
            }
        } else {
            s.bg_L = __fmaf_rn(s.bg_L, 0.85f, L * 0.15f);
            s.bg_a = __fmaf_rn(s.bg_a, 0.85f, a * 0.15f);
            s.bg_b = __fmaf_rn(s.bg_b, 0.85f, b * 0.15f);
            s.t = 0;
        }

        uint8_t dist_8 = __float2uint_rn(fminf(dist * 8.0f, 255.0f));
        output[pixel_idx] = dist_8;
        output[pixel_idx + 1] = dist_8;
        output[pixel_idx + 2] = dist_8;
    }
}

__global__ void morphological_opening_kernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ temp,
    uint8_t *__restrict__ output,
    int width, int height, int stride, int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int center_idx = y * stride + x * pixel_stride;
    
    uint8_t min_val = 255;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_idx = (y + dy) * stride + (x + dx) * pixel_stride;
            min_val = min(min_val, input[neighbor_idx]);
        }
    }
    temp[center_idx] = min_val;
    
    __syncthreads();
    
    uint8_t max_val = 0;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            int neighbor_idx = (y + dy) * stride + (x + dx) * pixel_stride;
            max_val = max(max_val, temp[neighbor_idx]);
        }
    }
    
    output[center_idx] = max_val;
    output[center_idx + 1] = max_val;
    output[center_idx + 2] = max_val;
}

__global__ void hysteresis_init_optimized(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ output,
    Point *__restrict__ queue,
    int *__restrict__ queue_size,
    int width, int height, int stride, int pixel_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    int y = idx / width;
    int x = idx % width;
    int pixel_idx = y * stride + x * pixel_stride;

    uint8_t val = input[pixel_idx];

    if (val >= HIGH_THRESHOLD) {
        output[pixel_idx] = 255;
        int pos = atomicAdd(queue_size, 1);
        if (pos < total_pixels) {
            queue[pos] = {x, y};
        }
    } else {
        output[pixel_idx] = 0;
    }
}

__global__ void apply_motion_mask_optimized(
    const uint8_t *__restrict__ original,
    const uint8_t *__restrict__ mask,
    uint8_t *__restrict__ output,
    int width, int height, int stride, int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * stride + x * pixel_stride;
    uint8_t mask_val = mask[idx];

    if (mask_val > 0) {
        output[idx] = (original[idx] + 255) >> 1;     // R
        output[idx + 1] = original[idx + 1] >> 1;     // G
        output[idx + 2] = original[idx + 2] >> 1;     // B
    } else {
        output[idx] = original[idx];
        output[idx + 1] = original[idx + 1];
        output[idx + 2] = original[idx + 2];
    }
}


extern "C"
{
    void filter_impl(uint8_t *h_buffer, int width, int height, int plane_stride, int pixel_stride)
    {
        static float total_time = 0.0f;
        static int frame_count = 0;
        static float min_time = 999999.0f;
        static float max_time = 0.0f;
        static auto start_benchmark = std::chrono::high_resolution_clock::now();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        size_t buffer_size = height * plane_stride;
        
        initialize_gpu_memory(width, height, plane_stride);

        CHECK_CUDA_ERROR(cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_original_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        dim3 block1d(256);
        dim3 grid1d((width * height + 255) / 256);

        background_estimation_kernel_optimized<<<grid, block>>>(
            d_buffer, d_temp_buffer1, d_states, width, height, plane_stride, pixel_stride);

        morphological_opening_kernel<<<grid, block>>>(
            d_temp_buffer1, d_temp_buffer2, d_mask_buffer, width, height, plane_stride, pixel_stride);

        CHECK_CUDA_ERROR(cudaMemset(d_queue_size, 0, sizeof(int)));
        hysteresis_init_optimized<<<grid1d, block1d>>>(
            d_mask_buffer, d_temp_buffer1, d_queue, d_queue_size, width, height, plane_stride, pixel_stride);

        apply_motion_mask_optimized<<<grid, block>>>(
            d_original_buffer, d_temp_buffer1, d_buffer, width, height, plane_stride, pixel_stride);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        total_time += milliseconds;
        frame_count++;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
        
        if (frame_count % 30 == 0) {
            float avg_time = total_time / frame_count;
            float avg_fps = 1000.0f / avg_time;
            float pixels_per_sec = (width * height * avg_fps) / 1000000.0f; // MPix/s
            
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_benchmark).count();
            
            std::cerr << "[CUDA-PERF] "
                      << "Resolution=" << width << "x" << height << " "
                      << "AvgFPS=" << std::fixed << std::setprecision(1) << avg_fps << " "
                      << "AvgTime=" << std::setprecision(2) << avg_time << "ms "
                      << "MinTime=" << min_time << "ms "
                      << "MaxTime=" << max_time << "ms "
                      << "MPix/s=" << std::setprecision(1) << pixels_per_sec << " "
                      << "Frames=" << frame_count << " "
                      << "Duration=" << duration << "s"
                      << std::endl;
            
            total_time = 0;
            frame_count = 0;
            min_time = 999999.0f;
            max_time = 0.0f;
            start_benchmark = now;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}