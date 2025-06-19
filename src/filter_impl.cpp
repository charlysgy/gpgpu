#include "filter_impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
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
                                     int plane_stride, int pixel_stride, u_int8_t *mask_buffer)
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

        // On stocke la distance dans le buffer monochrome pour les commandes SIMD
        auto dist_8 = static_cast<int>(dist);
        mask_buffer[y * width + x] = dist_8;
      }
    }
  }

  void mask_cleaning_process(uint8_t *buffer,
                             int width,
                             int height,
                             int plane_stride,
                             int pixel_stride)
  {
    const int radius = 3;
    int total = width * height;
    std::vector<uint8_t> eroded(total);

    for (int y = radius; y < height - radius; ++y)
    {
      uint8_t *row_m = buffer + (y - 1) * width;
      uint8_t *row = buffer + y * width;
      uint8_t *row_p = buffer + (y + 1) * width;
      uint8_t *out = eroded.data() + y * width;

      int x = radius;

      for (; x <= width - radius - 32; x += 32)
      {
        // Calcul le minimum des 3 pixels de la ligne précédente
        __m256i r0a = _mm256_loadu_si256((__m256i *)(row_m + x - 1));
        __m256i r0b = _mm256_loadu_si256((__m256i *)(row_m + x));
        __m256i r0c = _mm256_loadu_si256((__m256i *)(row_m + x + 1));
        __m256i m0 = _mm256_min_epu8(_mm256_min_epu8(r0a, r0b), r0c);

        // Calcul le minimum des 3 pixels de la ligne courante
        __m256i r1a = _mm256_loadu_si256((__m256i *)(row + x - 1));
        __m256i r1b = _mm256_loadu_si256((__m256i *)(row + x));
        __m256i r1c = _mm256_loadu_si256((__m256i *)(row + x + 1));
        __m256i m1 = _mm256_min_epu8(_mm256_min_epu8(r1a, r1b), r1c);

        // Calcul le minimum des 3 pixels de la ligne suivante
        __m256i r2a = _mm256_loadu_si256((__m256i *)(row_p + x - 1));
        __m256i r2b = _mm256_loadu_si256((__m256i *)(row_p + x));
        __m256i r2c = _mm256_loadu_si256((__m256i *)(row_p + x + 1));
        __m256i m2 = _mm256_min_epu8(_mm256_min_epu8(r2a, r2b), r2c);

        // Vertical min across the three horizontal minima
        __m256i m01 = _mm256_min_epu8(m0, m1);
        __m256i m_final = _mm256_min_epu8(m01, m2);
        _mm256_storeu_si256((__m256i *)(out + x), m_final);
      }
      // Scalar fallback for remaining pixels
      for (; x < width - radius; ++x)
      {
        uint8_t mv = 255;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            mv = std::min(mv, buffer[(y + dy) * width + (x + dx)]);
        out[x] = mv;
      }
    }

    // --- Dilation (max filter 3x3) ---
    for (int y = radius; y < height - radius; ++y)
    {
      uint8_t *row_m = eroded.data() + (y - 1) * width;
      uint8_t *row = eroded.data() + y * width;
      uint8_t *row_p = eroded.data() + (y + 1) * width;
      uint8_t *out = buffer + y * width;

      int x = radius;
      // Vectorized loop
      for (; x <= width - radius - 32; x += 32)
      {
        __m256i r0a = _mm256_loadu_si256((__m256i *)(row_m + x - 1));
        __m256i r0b = _mm256_loadu_si256((__m256i *)(row_m + x));
        __m256i r0c = _mm256_loadu_si256((__m256i *)(row_m + x + 1));
        __m256i m0 = _mm256_max_epu8(_mm256_max_epu8(r0a, r0b), r0c);

        __m256i r1a = _mm256_loadu_si256((__m256i *)(row + x - 1));
        __m256i r1b = _mm256_loadu_si256((__m256i *)(row + x));
        __m256i r1c = _mm256_loadu_si256((__m256i *)(row + x + 1));
        __m256i m1 = _mm256_max_epu8(_mm256_max_epu8(r1a, r1b), r1c);

        __m256i r2a = _mm256_loadu_si256((__m256i *)(row_p + x - 1));
        __m256i r2b = _mm256_loadu_si256((__m256i *)(row_p + x));
        __m256i r2c = _mm256_loadu_si256((__m256i *)(row_p + x + 1));
        __m256i m2 = _mm256_max_epu8(_mm256_max_epu8(r2a, r2b), r2c);

        __m256i m01 = _mm256_max_epu8(m0, m1);
        __m256i m_final = _mm256_max_epu8(m01, m2);
        _mm256_storeu_si256((__m256i *)(out + x), m_final);
      }
      // Scalar fallback
      for (; x < width - radius; ++x)
      {
        uint8_t mv = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx)
            mv = std::max(mv, eroded[(y + dy) * width + (x + dx)]);
        out[x] = mv;
      }
    }
  }

  void hysteresis_thresholding(uint8_t *buffer, int width, int height,
                               int plane_stride, int pixel_stride)
  {
    const uint8_t low = 4;
    const uint8_t high = 30;
    const int total = width * height;

    // Plutot que de faire un vecteur de pair on fait deux buffer (low et high)
    std::vector<uint8_t> mask_low(total), mask_high(total);
    __m256i v_low = _mm256_set1_epi8((char)(low - 1));
    __m256i v_high = _mm256_set1_epi8((char)(high - 1));
    int i = 0;
    for (; i <= total - 32; i += 32)
    {
      __m256i pix = _mm256_loadu_si256((__m256i *)(buffer + i));
      __m256i ml = _mm256_cmpgt_epi8(pix, v_low);
      __m256i mh = _mm256_cmpgt_epi8(pix, v_high);
      _mm256_storeu_si256((__m256i *)(mask_low.data() + i), ml);
      _mm256_storeu_si256((__m256i *)(mask_high.data() + i), mh);
    }
    for (; i < total; ++i)
    {
      mask_low[i] = buffer[i] > low - 1 ? 0xFF : 0x00;
      mask_high[i] = buffer[i] > high - 1 ? 0xFF : 0x00;
    }

    std::vector<uint8_t> tmp(mask_high);
    bool changed = true;
    while (changed)
    {
      changed = false;
      for (int y = 1; y < height - 1; ++y)
      {
        uint8_t *r0 = mask_high.data() + (y - 1) * width;
        uint8_t *r1 = mask_high.data() + y * width;
        uint8_t *r2 = mask_high.data() + (y + 1) * width;
        uint8_t *out = tmp.data() + y * width;
        int x = 1;
        for (; x <= width - 1 - 32; x += 32)
        {
          __m256i a0 = _mm256_loadu_si256((__m256i *)(r0 + x - 1));
          __m256i a1 = _mm256_loadu_si256((__m256i *)(r0 + x));
          __m256i a2 = _mm256_loadu_si256((__m256i *)(r0 + x + 1));
          __m256i m0 = _mm256_max_epu8(_mm256_max_epu8(a0, a1), a2);

          __m256i b0 = _mm256_loadu_si256((__m256i *)(r1 + x - 1));
          __m256i b1 = _mm256_loadu_si256((__m256i *)(r1 + x));
          __m256i b2 = _mm256_loadu_si256((__m256i *)(r1 + x + 1));
          __m256i m1 = _mm256_max_epu8(_mm256_max_epu8(b0, b1), b2);

          __m256i c0 = _mm256_loadu_si256((__m256i *)(r2 + x - 1));
          __m256i c1 = _mm256_loadu_si256((__m256i *)(r2 + x));
          __m256i c2 = _mm256_loadu_si256((__m256i *)(r2 + x + 1));
          __m256i m2 = _mm256_max_epu8(_mm256_max_epu8(c0, c1), c2);

          __m256i m01 = _mm256_max_epu8(m0, m1);
          __m256i max3 = _mm256_max_epu8(m01, m2);
          __m256i ml = _mm256_loadu_si256((__m256i *)(mask_low.data() + y * width + x));
          __m256i res = _mm256_and_si256(max3, ml);

          _mm256_storeu_si256((__m256i *)(out + x), res);
        }
        // fallback scalaire
        for (; x < width - 1; ++x)
        {
          uint8_t mv = 0;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              mv = std::max(mv,
                            mask_high[(y + dy) * width + (x + dx)]);
          tmp[y * width + x] = mv & mask_low[y * width + x];
        }
      }
      if (std::memcmp(tmp.data(), mask_high.data(), total) != 0)
      {
        changed = true;
        mask_high.swap(tmp);
      }
    }

    i = 0;
    for (; i <= total - 32; i += 32)
    {
      __m256i mh = _mm256_loadu_si256((__m256i *)(mask_high.data() + i));
      _mm256_storeu_si256((__m256i *)(buffer + i), mh);
    }
    for (; i < total; ++i)
    {
      buffer[i] = mask_high[i] ? 255 : 0;
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

    // On créer un nouveau buffer qui va accueillir le masque sous forme
    // de pixel monochrome [0, 255] donc on pourra utiliser les optimisations SIMD
    std::vector<uint8_t> mask_buffer(width * height);

    // Fait la transition entre l'image RGB non adaptée aux commandes SIMD
    // et le masque monochrome qui va être utilisé pour vaec les commandes SIMD
    background_estimation_process(buffer,
                                  width, height, plane_stride, pixel_stride, mask_buffer.data());

    // A partir d'ici on peut utiliser les commandes SIMD
    mask_cleaning_process(mask_buffer.data(),
                          width, height, plane_stride, pixel_stride);
    hysteresis_thresholding(mask_buffer.data(),
                            width, height, plane_stride, pixel_stride);

    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        int idx = y * plane_stride + x * pixel_stride;

        // Le masque est monochrome : n’importe quel canal suffit
        if (mask_buffer[y * width + x] > 0)
        {
          buffer[idx] = (buffer[idx] + 255) / 2;
        }
      }
    }
  }
}
