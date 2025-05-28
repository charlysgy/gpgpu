#include "filter_impl.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "logo.h"

struct rgb {
  uint8_t r, g, b;
};

/**
 * As explained in presentation slides of the project, here are the steps to
 * implement:
 * - Background estimation process
 * - Mask cleaning process
 * - - Noise removing
 * - - Hysteresis thresholding
 * - - Binary masking
 *
 * Each step has his own method with detailed docstring
 */
extern "C" {
/************ Let's first implement som helper methods ************************/
struct PixelState {
  float bg_L, bg_a, bg_b;
  float cand_L, cand_a, cand_b;
  int t;
  int initialized;
};

std::vector<struct PixelState> states{};

inline float srgb_to_linear(uint8_t c) {
  float fc = c / 255.0f;
  return (fc <= 0.04045f) ? fc / 12.92f : powf((fc + 0.055f) / 1.055f, 2.4f);
}

inline void rgb_to_xyz(uint8_t R, uint8_t G, uint8_t B, float& X, float& Y, float& Z) {
  float r = srgb_to_linear(R);
  float g = srgb_to_linear(G);
  float b = srgb_to_linear(B);

  X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
  Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
  Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
}

inline float f_xyz(float t) {
  return (t > 0.008856f) ? powf(t, 1.0f / 3.0f) : (7.787f * t + 16.0f / 116.0f);
}

inline void xyz_to_lab(float X, float Y, float Z, float& L, float& a, float& b) {
  const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
  float fx = f_xyz(X / Xn);
  float fy = f_xyz(Y / Yn);
  float fz = f_xyz(Z / Zn);
  L = 116.0f * fy - 16.0f;
  a = 500.0f * (fx - fy);
  b = 200.0f * (fy - fz);
}

inline void rgb_to_lab(uint8_t R, uint8_t G, uint8_t B, float& L, float& a, float& b) {
  float X, Y, Z;
  rgb_to_xyz(R, G, B, X, Y, Z);
  xyz_to_lab(X, Y, Z, L, a, b);
}

/******************************************************************************/
void background_estimation_process(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride);
void filter_impl(uint8_t *buffer, int width, int height, int plane_stride,
                 int pixel_stride) {
  //        for (int y = 0; y < height; ++y)
  //        {
  //            rgb* lineptr = (rgb*) (buffer + y * plane_stride);
  //            for (int x = 0; x < width; ++x)
  //            {
  //                lineptr[x].r = 0; // Back out red component
  //
  //                if (x < logo_width && y < logo_height)
  //                {
  //                    float alpha = logo_data[y * logo_width + x] / 255.f;
  //                    lineptr[x].r = uint8_t(alpha * lineptr[x].r + (1-alpha)
  //                    * 255); lineptr[x].g = uint8_t(alpha * lineptr[x].g +
  //                    (1-alpha) * 255); lineptr[x].b = uint8_t(alpha *
  //                    lineptr[x].b + (1-alpha) * 255);
  //
  //                }
  //            }
  //        }
  states.reserve(width * height);
  background_estimation_process(buffer, width, height, plane_stride, pixel_stride);
}

void background_estimation_process(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride) {
  for (int y = 0; y < height; ++y) {
    rgb* lineptr = (rgb*) (buffer + y * plane_stride);
    for (int x = 0; x < width; ++x) {
      uint8_t R = lineptr[x].r;
      uint8_t G = lineptr[x].g;
      uint8_t B = lineptr[x].b;

      float L, a, b;
      rgb_to_lab(R, G, B, L, a, b);

      PixelState& state = states[y * width + x];

      if (!state.initialized) {
        state.bg_L = L;
        state.bg_a = a;
        state.bg_b = b;
        state.initialized = true;
        state.t = 0;
        continue;
      }

      float dL = state.bg_L - L;
      float da = state.bg_a - a;
      float db = state.bg_b - b;
      float dist = std::sqrt(dL * dL + da * da + db * db);

      bool match = dist < 25.0f;

      if (!match) {
        if (state.t == 0) {
          state.cand_L = L;
          state.cand_a = a;
          state.cand_b = b;
          state.t++;
        } else if (state.t < 100) {
          state.cand_L = (state.cand_L + L) / 2.0f;
          state.cand_a = (state.cand_a + a) / 2.0f;
          state.cand_b = (state.cand_b + b) / 2.0f;
          state.t++;
        } else {
          std::swap(state.bg_L, state.cand_L);
          std::swap(state.bg_a, state.cand_a);
          std::swap(state.bg_b, state.cand_b);
          state.t = 0;
        }
      } else {
        state.bg_L = (state.bg_L + L) / 2.0f;
        state.bg_a = (state.bg_a + a) / 2.0f;
        state.bg_b = (state.bg_b + b) / 2.0f;
        state.t = 0;
      }

      // Tu peux stocker `dist` ici si besoin (dans un masque par ex)
      auto dist_8 = static_cast<u_int8_t>(dist * 2.5);
      lineptr[x].r = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
      lineptr[x].g = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
      lineptr[x].b = dist_8 > 255 ? 255 : (dist_8 < 0 ? 0 : dist_8);
    }
  }
}


void mask_cleaning_process(uint8_t *buffer, int width, int height,
                           int plane_stride, int pixel_stride) {}
}
