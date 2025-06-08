#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include "filter_impl.h"
#include "logo.h"

// =============================================================================
// CONSTANTS AND MACROS
// =============================================================================

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

const uint8_t LOW_THRESHOLD = 4;
const uint8_t HIGH_THRESHOLD = 30;

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

struct rgb
{
    uint8_t r, g, b;
};

struct PixelState
{
    float bg_L, bg_a, bg_b; // Background LAB values
    float cand_L, cand_a, cand_b; // Candidate LAB values
    int t; // Time counter
    int initialized; // Initialization flag
};

// =============================================================================
// GLOBAL DEVICE MEMORY
// =============================================================================

__constant__ uint8_t *logo;

// Static variables for pixel state management
static PixelState *d_states = nullptr;
static size_t image_size = 0;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        std::exit(EXIT_FAILURE);
    }
}

namespace
{
    void load_logo()
    {
        static auto buffer =
            std::unique_ptr<std::byte, decltype(&cudaFree)>{ nullptr,
                                                             &cudaFree };

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte *ptr;

            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height,
                             cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }
    }
} // anonymous namespace

// =============================================================================
// DEVICE UTILITY FUNCTIONS
// =============================================================================

__device__ float srgb_to_linear(float c)
{
    return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ void rgb_to_lab(uint8_t R, uint8_t G, uint8_t B, float &L, float &a,
                           float &b)
{
    // Convert sRGB to linear RGB
    float r = srgb_to_linear(R / 255.0f);
    float g = srgb_to_linear(G / 255.0f);
    float b_ = srgb_to_linear(B / 255.0f);

    // Linear RGB to XYZ conversion
    float X = r * 0.4124564f + g * 0.3575761f + b_ * 0.1804375f;
    float Y = r * 0.2126729f + g * 0.7151522f + b_ * 0.0721750f;
    float Z = r * 0.0193339f + g * 0.1191920f + b_ * 0.9503041f;

    // XYZ to LAB conversion
    float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
    float fx = (X / Xn > 0.008856f) ? cbrtf(X / Xn)
                                    : (7.787f * (X / Xn) + 16.0f / 116.0f);
    float fy = (Y / Yn > 0.008856f) ? cbrtf(Y / Yn)
                                    : (7.787f * (Y / Yn) + 16.0f / 116.0f);
    float fz = (Z / Zn > 0.008856f) ? cbrtf(Z / Zn)
                                    : (7.787f * (Z / Zn) + 16.0f / 116.0f);

    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}

// =============================================================================
// CUDA KERNELS
// =============================================================================

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer Video buffer
/// @param width Image width
/// @param height Image height
/// @param stride Row stride in bytes
__global__ void remove_red_channel_inp(std::byte *buffer, int width, int height,
                                       int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    rgb *lineptr = (rgb *)(buffer + y * stride);

    if (y < logo_height && x < logo_width)
    {
        float alpha = logo[y * logo_width + x] / 255.0f;
        lineptr[x].r = 0;
        lineptr[x].g =
            static_cast<uint8_t>(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b =
            static_cast<uint8_t>(alpha * lineptr[x].b + (1 - alpha) * 255);
    }
    else
    {
        lineptr[x].r = 0;
    }
}

/// @brief Apply motion mask to original image with red channel enhancement
__global__ void apply_motion_mask_kernel(uint8_t *original_buffer,
                                         const uint8_t *mask_buffer, int width,
                                         int height, int stride,
                                         int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * stride + x * pixel_stride;

    // Check if this pixel has motion (mask is non-zero)
    uint8_t mask_val = mask_buffer[idx];

    if (mask_val > 0) // Motion detected
    {
        original_buffer[idx] = (original_buffer[idx] + 255) / 2; // R
        original_buffer[idx + 1] = original_buffer[idx + 1] / 2; // G
        original_buffer[idx + 2] = original_buffer[idx + 2] / 2; // B
    }
    // If no motion, leave original pixel unchanged
}

/// @brief Background estimation kernel using LAB color space
__global__ void background_estimation_kernel(uint8_t *rgb, PixelState *states,
                                             int width, int height, int stride,
                                             int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Get current pixel RGB values
    int index = y * stride + x * pixel_stride;
    uint8_t R = rgb[index];
    uint8_t G = rgb[index + 1];
    uint8_t B = rgb[index + 2];

    // Convert to LAB color space
    float L, a, b;
    rgb_to_lab(R, G, B, L, a, b);

    // Get pixel state
    int idx = y * width + x;
    PixelState &s = states[idx];

    if (!s.initialized)
    {
        // Initialize background with first frame
        s.bg_L = L;
        s.bg_a = a;
        s.bg_b = b;
        s.t = 0;
        s.initialized = 1;
    }
    else
    {
        // Calculate distance between background and current frame
        float dL = s.bg_L - L;
        float da = s.bg_a - a;
        float db = s.bg_b - b;
        float dist = sqrtf(dL * dL + da * da + db * db);
        bool match = dist < 20.0f;

        if (!match)
        {
            // Pixel doesn't match background
            if (s.t == 0)
            {
                // Start new candidate
                s.cand_L = L;
                s.cand_a = a;
                s.cand_b = b;
                s.t++;
            }
            else if (s.t < 50)
            {
                // Update candidate with running average
                s.cand_L = (s.cand_L * 0.5f + L * 0.5f);
                s.cand_a = (s.cand_a * 0.5f + a * 0.5f);
                s.cand_b = (s.cand_b * 0.5f + b * 0.5f);
                s.t++;
            }
            else
            {
                // Accept candidate as new background
                s.bg_L = s.cand_L;
                s.bg_a = s.cand_a;
                s.bg_b = s.cand_b;
                s.t = 0;
            }
        }
        else
        {
            // Pixel matches background - update with weighted average
            s.bg_L = (s.bg_L * 0.8f + L * 0.2f);
            s.bg_a = (s.bg_a * 0.8f + a * 0.2f);
            s.bg_b = (s.bg_b * 0.8f + b * 0.2f);
            s.t = 0;
        }

        // Output distance scaled to [0, 255]
        rgb[index] =
            static_cast<uint8_t>(fminf(fmaxf(dist * 2.55f, 0.0f), 255.0f));
    }
}

/// @brief Morphological erosion kernel
__global__ void erosion_kernel(uint8_t *input, uint8_t *output, int width,
                               int height, int stride, int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    uint8_t min_val = 255;

    // 3x3 neighborhood
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            int idx = ny * stride + nx * pixel_stride;
            min_val = min(min_val, input[idx]);
        }
    }

    // Set all RGB channels to the same value
    int out_idx = y * stride + x * pixel_stride;
    output[out_idx] = min_val;
    output[out_idx + 1] = min_val;
    output[out_idx + 2] = min_val;
}

/// @brief Morphological dilation kernel
__global__ void dilation_kernel(uint8_t *input, uint8_t *output, int width,
                                int height, int stride, int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    uint8_t max_val = 0;

    // 3x3 neighborhood
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            int idx = ny * stride + nx * pixel_stride;
            max_val = max(max_val, input[idx]);
        }
    }

    // Set all RGB channels to the same value
    int out_idx = y * stride + x * pixel_stride;
    output[out_idx] = max_val;
    output[out_idx + 1] = max_val;
    output[out_idx + 2] = max_val;
}

/// @brief Initialize hysteresis thresholding
__global__ void init_hysteresis_kernel(const uint8_t *input, uint8_t *current,
                                       int width, int height, int stride,
                                       int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * stride + x * pixel_stride;
    current[idx] = (input[idx] >= HIGH_THRESHOLD) ? 255 : 0;
}

/// @brief Propagate hysteresis thresholding
__global__ void propagate_hysteresis_kernel(const uint8_t *input,
                                            const uint8_t *prev, uint8_t *next,
                                            int width, int height, int stride,
                                            int pixel_stride, bool *changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
        return;

    int idx = y * stride + x * pixel_stride;

    // Skip if already processed or outside threshold range
    if (prev[idx] == 255 || input[idx] < LOW_THRESHOLD
        || input[idx] >= HIGH_THRESHOLD)
        return;

    // Check 8-connected neighbors
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            int nidx = ny * stride + nx * pixel_stride;

            if (prev[nidx] == 255)
            {
                next[idx] = 255;
                *changed = true;
                return;
            }
        }
    }
}

/// @brief Finalize hysteresis mask by copying to all RGB channels
__global__ void finalize_hysteresis_mask(uint8_t *buffer, const uint8_t *mask,
                                         int width, int height, int stride,
                                         int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * stride + x * pixel_stride;
    uint8_t val = mask[idx];
    buffer[idx] = val;
    buffer[idx + 1] = val;
    buffer[idx + 2] = val;
}

// =============================================================================
// CUDA WRAPPER FUNCTIONS
// =============================================================================

void cuda_hysteresis(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride)
{
    size_t size = height * stride;

    // Allocate device memory
    uint8_t *d_input, *d_prev, *d_next;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_prev, size);
    cudaMalloc(&d_next, size);
    cudaMemcpy(d_input, buffer, size, cudaMemcpyHostToDevice);

    // Setup grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Initialize with high threshold
    init_hysteresis_kernel<<<grid, block>>>(d_input, d_prev, width, height,
                                            stride, pixel_stride);

    // Allocate flag for iteration control
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // Iterative propagation (max 100 iterations)
    for (int iter = 0; iter < 100; ++iter)
    {
        bool changed = false;
        cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice);

        propagate_hysteresis_kernel<<<grid, block>>>(d_input, d_prev, d_next,
                                                     width, height, stride,
                                                     pixel_stride, d_changed);

        std::swap(d_prev, d_next);

        cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!changed)
            break;
    }

    // Finalize and copy back
    finalize_hysteresis_mask<<<grid, block>>>(d_input, d_prev, width, height,
                                              stride, pixel_stride);
    cudaMemcpy(buffer, d_input, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_prev);
    cudaFree(d_next);
    cudaFree(d_changed);
}

void cuda_opening(uint8_t *buffer, int width, int height, int stride,
                  int pixel_stride)
{
    size_t buffer_size = height * stride;

    // Allocate device memory
    uint8_t *d_input, *d_eroded, *d_opened;
    cudaMalloc(&d_input, buffer_size);
    cudaMalloc(&d_eroded, buffer_size);
    cudaMalloc(&d_opened, buffer_size);

    cudaMemcpy(d_input, buffer, buffer_size, cudaMemcpyHostToDevice);

    // Setup grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // Perform opening (erosion followed by dilation)
    erosion_kernel<<<grid, block>>>(d_input, d_eroded, width, height, stride,
                                    pixel_stride);
    dilation_kernel<<<grid, block>>>(d_eroded, d_opened, width, height, stride,
                                     pixel_stride);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(buffer, d_opened, buffer_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_eroded);
    cudaFree(d_opened);
}

// =============================================================================
// MAIN FILTER IMPLEMENTATION
// =============================================================================

extern "C"
{
    void filter_impl(uint8_t *h_buffer, int width, int height, int plane_stride,
                     int pixel_stride)
    {
        // Create a copy of the original buffer to preserve it
        size_t buffer_size = height * plane_stride;
        uint8_t *h_original_buffer = (uint8_t *)malloc(buffer_size);
        memcpy(h_original_buffer, h_buffer, buffer_size);

        // Initialize pixel states on first call
        if (!d_states)
        {
            image_size = width * height;
            cudaMalloc(&d_states, image_size * sizeof(PixelState));
            cudaMemset(d_states, 0, image_size * sizeof(PixelState));
        }

        // Allocate device buffer and copy input data
        uint8_t *d_buffer;
        cudaMalloc(&d_buffer, height * plane_stride);
        cudaMemcpy(d_buffer, h_buffer, height * plane_stride,
                   cudaMemcpyHostToDevice);

        // Setup grid and block dimensions
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        // Apply background estimation
        background_estimation_kernel<<<grid, block>>>(
            d_buffer, d_states, width, height, plane_stride, pixel_stride);

        // Copy intermediate result back to host for morphological operations
        cudaMemcpy(h_buffer, d_buffer, height * plane_stride,
                   cudaMemcpyDeviceToHost);

        // Apply morphological opening
        cuda_opening(h_buffer, width, height, plane_stride, pixel_stride);

        // Apply hysteresis thresholding
        cuda_hysteresis(h_buffer, width, height, plane_stride, pixel_stride);

        // Apply the mask to the original image with red channel enhancement
        uint8_t *d_original_buffer, *d_final_mask;
        cudaMalloc(&d_original_buffer, buffer_size);
        cudaMalloc(&d_final_mask, buffer_size);

        cudaMemcpy(d_original_buffer, h_original_buffer, buffer_size,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_final_mask, h_buffer, buffer_size, cudaMemcpyHostToDevice);

        // Apply motion mask with red highlighting
        apply_motion_mask_kernel<<<grid, block>>>(d_original_buffer,
                                                  d_final_mask, width, height,
                                                  plane_stride, pixel_stride);
        // Copy the final result back to host
        cudaMemcpy(h_buffer, d_original_buffer, buffer_size,
                   cudaMemcpyDeviceToHost);
        // Cleanup
        cudaFree(d_buffer);
        cudaFree(d_original_buffer);
        cudaFree(d_final_mask);
        free(h_original_buffer);
    }

} // extern "C"
