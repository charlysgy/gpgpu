#include "filter_impl.h"

#include "logo.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
    std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
    // We don't exit when we encounter CUDA errors in this example.
    std::exit(EXIT_FAILURE);
  }
}

struct rgb {
  uint8_t r, g, b;
};

struct PixelState {
  float bg_L, bg_a, bg_b;
  float cand_L, cand_a, cand_b;
  int t;
  int initialized;
};

__device__ float srgb_to_linear(float c) {
  return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ void rgb_to_lab(uint8_t R, uint8_t G, uint8_t B, float &L, float &a,
                           float &b) {
  float r = srgb_to_linear(R / 255.0f);
  float g = srgb_to_linear(G / 255.0f);
  float b_ = srgb_to_linear(B / 255.0f);

  float X = r * 0.4124564f + g * 0.3575761f + b_ * 0.1804375f;
  float Y = r * 0.2126729f + g * 0.7151522f + b_ * 0.0721750f;
  float Z = r * 0.0193339f + g * 0.1191920f + b_ * 0.9503041f;

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

__constant__ uint8_t *logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void remove_red_channel_inp(std::byte *buffer, int width, int height,
                                       int stride) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width || y >= height)
    return;

  rgb *lineptr = (rgb *)(buffer + y * stride);
  if (y < logo_height && x < logo_width) {
    float alpha = logo[y * logo_width + x] / 255.f;
    lineptr[x].r = 0;
    lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
    lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
  } else {
    lineptr[x].r = 0;
  }
}

namespace {
void load_logo() {
  static auto buffer =
      std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree};

  if (buffer == nullptr) {
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
} // namespace

extern "C" {
__global__ void background_estimation_kernel(uint8_t *rgb, PixelState *states,
                                             int width, int height, int stride,
                                             int pixel_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int index = y * stride + x * pixel_stride;
  uint8_t R = rgb[index];
  uint8_t G = rgb[index + 1];
  uint8_t B = rgb[index + 2];

  float L, a, b;
  rgb_to_lab(R, G, B, L, a, b);

  int idx = y * width + x;
  PixelState &s = states[idx];

  if (!s.initialized) {
    s.bg_L = L;
    s.bg_a = a;
    s.bg_b = b;
    s.t = 0;
    s.initialized = 1;
  } else {
    // Calcul la distance entre le background et la frame courante
    float dL = s.bg_L - L;
    float da = s.bg_a - a;
    float db = s.bg_b - b;
    float dist = sqrtf(dL * dL + da * da + db * db);
    bool match = dist < 20.0f;

    if (!match) {
      if (s.t == 0) {
        s.cand_L = L;
        s.cand_a = a;
        s.cand_b = b;
        ++(s.t);
      } else if (s.t < 50) {
        s.cand_L = (s.cand_L * 0.5 + L * 0.5);
        s.cand_a = (s.cand_a * 0.5 + a * 0.5);
        s.cand_b = (s.cand_b * 0.5 + b * 0.5);
        ++(s.t);
      } else {
        s.bg_L = s.cand_L;
        s.bg_a = s.cand_a;
        s.bg_b = s.cand_b;
        s.t = 0;
      }
    } else {
      // On pondère les valeurs pour que le background ait plus de poids
      // que les perturbations de la frame courante
      s.bg_L = (s.bg_L * 0.8 + L * 0.2);
      s.bg_a = (s.bg_a * 0.8 + a * 0.2);
      s.bg_b = (s.bg_b * 0.8 + b * 0.2);
      s.t = 0;
    }

    rgb[index] = static_cast<uint8_t>(fminf(fmaxf(dist * 2.55f, 0.0f), 255.0f));
  }
}

__global__ void erosion_kernel(uint8_t *input, uint8_t *output, int width,
                               int height, int stride, int pixel_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  uint8_t min_val = 255;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int nx = x + dx;
      int ny = y + dy;
      int idx = ny * stride + nx * pixel_stride;
      min_val = min(min_val, input[idx]);
    }
  }

  int out_idx = y * stride + x * pixel_stride;
  output[out_idx] = min_val;
  output[out_idx + 1] = min_val;
  output[out_idx + 2] = min_val;
}

__global__ void dilation_kernel(uint8_t *input, uint8_t *output, int width,
                                int height, int stride, int pixel_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  uint8_t max_val = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int nx = x + dx;
      int ny = y + dy;
      int idx = ny * stride + nx * pixel_stride;
      max_val = max(max_val, input[idx]);
    }
  }

  int out_idx = y * stride + x * pixel_stride;
  output[out_idx] = max_val;
  output[out_idx + 1] = max_val;
  output[out_idx + 2] = max_val;
}

const u_int8_t LOW_THRESHOLD = 80;
const u_int8_t HIGH_THRESHOLD = 200;

__global__ void init_hysteresis_kernel(const uint8_t *input, uint8_t *current,
                                       int width, int height, int stride,
                                       int pixel_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * stride + x * pixel_stride;
  current[idx] = (input[idx] >= HIGH_THRESHOLD) ? 255 : 0;
}

__global__ void propagate_hysteresis_kernel(const uint8_t *input,
                                            const uint8_t *prev, uint8_t *next,
                                            int width, int height, int stride,
                                            int pixel_stride, bool *changed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  int idx = y * stride + x * pixel_stride;
  if (prev[idx] == 255 || input[idx] < LOW_THRESHOLD ||
      input[idx] >= HIGH_THRESHOLD)
    return;

  // Check 8 neighbors
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int nx = x + dx, ny = y + dy;
      int nidx = ny * stride + nx * pixel_stride;
      if (prev[nidx] == 255) {
        next[idx] = 255;
        *changed = true;
        return;
      }
    }
  }
}

__global__ void finalize_hysteresis_mask(uint8_t *buffer, const uint8_t *mask,
                                         int width, int height, int stride,
                                         int pixel_stride) {
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

void cuda_hysteresis(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride) {
  size_t size = height * stride;

  uint8_t *d_input, *d_prev, *d_next;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_prev, size);
  cudaMalloc(&d_next, size);
  cudaMemcpy(d_input, buffer, size, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);

  init_hysteresis_kernel<<<grid, block>>>(d_input, d_prev, width, height,
                                          stride, pixel_stride);

  bool *d_changed;
  cudaMalloc(&d_changed, sizeof(bool));

  for (int iter = 0; iter < 100; ++iter) { // cap sur 100 itérations max
    bool changed = false;
    cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice);

    propagate_hysteresis_kernel<<<grid, block>>>(d_input, d_prev, d_next, width,
                                                 height, stride, pixel_stride,
                                                 d_changed);

    std::swap(d_prev, d_next);

    cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    if (!changed)
      break;
  }

  finalize_hysteresis_mask<<<grid, block>>>(d_input, d_prev, width, height,
                                            stride, pixel_stride);
  cudaMemcpy(buffer, d_input, size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_prev);
  cudaFree(d_next);
  cudaFree(d_changed);
}

void cuda_opening(uint8_t *buffer, int width, int height, int stride,
                  int pixel_stride) {
  uint8_t *d_input, *d_eroded, *d_opened;
  size_t buffer_size = height * stride;

  cudaMalloc(&d_input, buffer_size);
  cudaMalloc(&d_eroded, buffer_size);
  cudaMalloc(&d_opened, buffer_size);

  cudaMemcpy(d_input, buffer, buffer_size, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);

  erosion_kernel<<<grid, block>>>(d_input, d_eroded, width, height, stride,
                                  pixel_stride);
  dilation_kernel<<<grid, block>>>(d_eroded, d_opened, width, height, stride,
                                   pixel_stride);

  cudaDeviceSynchronize();

  // Répliquer le résultat dans les canaux R, G, B
  dilation_kernel<<<grid, block>>>(d_eroded, d_opened, width, height, stride,
                                   pixel_stride); // output is grayscale
  cudaMemcpy(buffer, d_opened, buffer_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_eroded);
  cudaFree(d_opened);
}

// Array contenant les états des pixels pendant le traitement
static PixelState *d_states = nullptr;
// Taille de l'image (fixe pendant tout le traitement
static size_t image_size = 0;

void filter_impl(uint8_t *h_buffer, int width, int height, int plane_stride,
                 int pixel_stride) {

  // Au premier appel de filter_impl, l'array n'est pas alloué
  if (!d_states) {
    image_size = width * height;
    cudaMalloc(&d_states, image_size * sizeof(PixelState));
    cudaMemset(d_states, 0, image_size * sizeof(PixelState));
  }

  // Buffer stocké sur le device qui est une copie du buffer de pixel en input
  uint8_t *d_buffer;
  cudaMalloc(&d_buffer, height * plane_stride);
  cudaMemcpy(d_buffer, h_buffer, height * plane_stride, cudaMemcpyHostToDevice);

  // Définition du nombre de warp nécessaire sur le device
  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);

  background_estimation_kernel<<<grid, block>>>(d_buffer, d_states, width, height, plane_stride, pixel_stride);
  cudaMemcpy(h_buffer, d_buffer, height * plane_stride, cudaMemcpyDeviceToHost);
  cuda_opening(h_buffer, width, height, plane_stride, pixel_stride);
  cuda_hysteresis(h_buffer, width, height, plane_stride, pixel_stride);
  cudaFree(d_buffer);
}
}
