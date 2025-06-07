#include "filter_impl.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "logo.h"

/**
 * As explained in presentation slides of the project, here are the steps to
 * implement:
 * - Background estimation process
 * - Mask cleaning process
 * - - Noise removing
 * - - Hysteresis thresholding
 * - - Binary masking
 *
 * Each step has it's own method with detailed docstring
 */
extern "C"
{
  /************ Let's first implement some helper methods ************************/

  // Struct that hold RGB values of a pixel to ease buffer reading and writing
  struct rgb
  {
    uint8_t r, g, b;
  };

  // Struct that holds the state of each pixel during the background estimation process
  struct PixelState
  {
    float bg_L, bg_a, bg_b;       // Background color in Lab color space
    float cand_L, cand_a, cand_b; // Candidate color in Lab color space
    int t;                        // Time counter for candidate color
    int initialized;              // Flag to check if the pixel state has been initialized
  };

  inline float srgb_to_linear(uint8_t c)
  {
    float fc = c / 255.0f;
    return (fc <= 0.04045f) ? fc / 12.92f : powf((fc + 0.055f) / 1.055f, 2.4f);
  }

  inline void rgb_to_xyz(uint8_t R, uint8_t G, uint8_t B, float &X, float &Y,
                         float &Z)
  {
    float r = srgb_to_linear(R);
    float g = srgb_to_linear(G);
    float b = srgb_to_linear(B);

    X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
  }

  inline float f_xyz(float t)
  {
    return (t > 0.008856f) ? powf(t, 1.0f / 3.0f) : (7.787f * t + 16.0f / 116.0f);
  }

  inline void xyz_to_lab(float X, float Y, float Z, float &L, float &a,
                         float &b)
  {
    const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
    float fx = f_xyz(X / Xn);
    float fy = f_xyz(Y / Yn);
    float fz = f_xyz(Z / Zn);
    L = 116.0f * fy - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
  }

  inline void rgb_to_lab(uint8_t R, uint8_t G, uint8_t B, float &L, float &a,
                         float &b)
  {
    float X, Y, Z;
    rgb_to_xyz(R, G, B, X, Y, Z);
    xyz_to_lab(X, Y, Z, L, a, b);
  }

  /******************************************************************************/

  // Vector to hold the state of each pixel
  std::vector<struct PixelState> states{};

  // Lab ditance threshold for background estimation
  const float LAB_DISTANCE_THRESHOLD = 20.0f;

  void background_estimation_process(uint8_t *buffer, int width, int height,
                                     int plane_stride, int pixel_stride)
  {
    static int max_value = 0;
    for (int y = 0; y < height; ++y)
    {
      // Access the rgb value of each pixel in the current line
      struct rgb *lineptr = (struct rgb *)(buffer + y * plane_stride);
      for (int x = 0; x < width; ++x)
      {
        uint8_t R = lineptr[x].r;
        uint8_t G = lineptr[x].g;
        uint8_t B = lineptr[x].b;

        float L, a, b;
        rgb_to_lab(R, G, B, L, a, b);

        // Retrieve or initialize the pixel state
        PixelState &state = states[y * width + x];

        if (!state.initialized)
        {
          state.bg_L = L;
          state.bg_a = a;
          state.bg_b = b;
          state.initialized = true;
          state.t = 0;
          continue;
        }

        // Compute the distance between the current pixel color and the background color
        float dL = state.bg_L - L;
        float da = state.bg_a - a;
        float db = state.bg_b - b;
        float dist = std::sqrt(dL * dL + da * da + db * db);

        bool match = dist < LAB_DISTANCE_THRESHOLD;

        if (!match)
        {
          // If the pixel does not match the background and the candidate color is not initialized,
          if (state.t == 0)
          {
            state.cand_L = L;
            state.cand_a = a;
            state.cand_b = b;
            state.t++;
          }
          // Else, if the candidate color is initialized, we update it
          // and increment the time counter until it reaches 25 frames
          // that is the average framerate for standard videos
          else if (state.t < 25)
          {
            state.cand_L = (state.cand_L + L) / 2.0f;
            state.cand_a = (state.cand_a + a) / 2.0f;
            state.cand_b = (state.cand_b + b) / 2.0f;
            state.t++;
          }
          // If the time counter reaches 25 frames, we swap the background and candidate colors
          else
          {
            std::swap(state.bg_L, state.cand_L);
            std::swap(state.bg_a, state.cand_a);
            std::swap(state.bg_b, state.cand_b);
            state.t = 0;
          }
        }
        // If the pixel matches the background, we update the background color
        // by averaging the current pixel color with the background color
        else
        {
          state.bg_L = (state.bg_L * 1.7f + L * 0.3f) / 2.0f;
          state.bg_a = (state.bg_a * 1.7f + a * 0.3f) / 2.0f;
          state.bg_b = (state.bg_b * 1.7f + b * 0.3f) / 2.0f;
          state.t = 0;
        }

        max_value = std::max(max_value, static_cast<int>(dist));

        auto dist_8 = static_cast<int>(dist);
        lineptr[x].r = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
        lineptr[x].g = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
        lineptr[x].b = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
      }
    }
  }

  void mask_cleaning_process(uint8_t *buffer, int width, int height,
                             int plane_stride, int pixel_stride)
  {
    std::vector<uint8_t> eroded(height * plane_stride, 0);
    std::vector<uint8_t> opened(height * plane_stride, 0);

    int radius = 3;

    // Erosion
    for (int y = radius; y < height - radius; ++y)
    {
      for (int x = radius; x < width - radius; ++x)
      {
        uint8_t min_val = 255;
        for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dx = -1; dx <= 1; ++dx)
          {
            int nx = x + dx;
            int ny = y + dy;
            int idx = ny * plane_stride + nx * pixel_stride;
            min_val = std::min(min_val, buffer[idx]);
          }
        }
        int out_idx = y * plane_stride + x * pixel_stride;
        eroded[out_idx] = min_val;
      }
    }

    // --- Dilatation (max dans le disque) ---
    for (int y = radius; y < height - radius; ++y)
    {
      for (int x = radius; x < width - radius; ++x)
      {
        uint8_t max_val = 0;
        for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dx = -1; dx <= 1; ++dx)
          {
            int nx = x + dx;
            int ny = y + dy;
            int idx = ny * plane_stride + nx * pixel_stride;
            max_val = std::max(max_val, eroded[idx]);
          }
        }
        int out_idx = y * plane_stride + x * pixel_stride;
        opened[out_idx] = max_val;
      }
    }

    // --- Copier le résultat final dans le buffer original ---
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        int idx = y * plane_stride + x * pixel_stride;
        buffer[idx] = opened[idx];
        buffer[idx + 1] = opened[idx];
        buffer[idx + 2] = opened[idx];
      }
    }
  }

  void hysteresis_thresholding(uint8_t *buffer, int width, int height,
                               int plane_stride, int pixel_stride)
  {
    const uint8_t low = 4;   // seuil bas
    const uint8_t high = 30; // seuil haut

    std::vector<uint8_t> visited(height * plane_stride, 0);
    std::vector<std::pair<int, int>> stack;

    auto at = [&](int y, int x) -> uint8_t &
    {
      return buffer[y * plane_stride + x * pixel_stride];
    };

    // Étape 1 : initialiser les "forts" pixels et les empiler
    for (int y = 1; y < height - 1; ++y)
    {
      for (int x = 1; x < width - 1; ++x)
      {
        if (at(y, x) >= high)
        {
          visited[y * plane_stride + x * pixel_stride] = 1;
          stack.emplace_back(y, x);
        }
      }
    }

    // Étape 2 : propager les forts vers les moyens
    while (!stack.empty())
    {
      auto [y, x] = stack.back();
      stack.pop_back();

      for (int dy = -1; dy <= 1; ++dy)
      {
        for (int dx = -1; dx <= 1; ++dx)
        {
          int ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= height || nx < 0 || nx >= width)
            continue;

          int idx = ny * plane_stride + nx * pixel_stride;
          if (!visited[idx] && buffer[idx] >= low)
          {
            visited[idx] = 1;
            stack.emplace_back(ny, nx);
          }
        }
      }
    }

    // Étape 3 : suppression des pixels faibles non connectés
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        int idx = y * plane_stride + x * pixel_stride;
        uint8_t val = visited[idx] ? 255 : 0;

        // RGB = même valeur (grayscale)
        for (int c = 0; c < pixel_stride; ++c)
        {
          buffer[idx + c] = val;
        }
      }
    }
  }

  void filter_impl(uint8_t *buffer,
                   int width,
                   int height,
                   int plane_stride,
                   int pixel_stride)
  {
    if (states.size() != width * height)
      states.reserve(width * height);

    std::vector<uint8_t> mask_buffer(height * plane_stride);
    std::memcpy(mask_buffer.data(), buffer, height * plane_stride);

    background_estimation_process(mask_buffer.data(),
                                  width, height, plane_stride, pixel_stride);
    mask_cleaning_process(mask_buffer.data(),
                          width, height, plane_stride, pixel_stride);
    hysteresis_thresholding(mask_buffer.data(),
                            width, height, plane_stride, pixel_stride);
    mask_cleaning_process(mask_buffer.data(),
                          width, height, plane_stride, pixel_stride);

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        int idx = y * plane_stride + x * pixel_stride;

        // Le masque est monochrome : n’importe quel canal suffit
        uint8_t m = mask_buffer[idx];
        if (m) // pixel « mouvement »
        {
          buffer[idx] = (buffer[idx] + 255) / 2;
        }
      }
    }
  }
}
