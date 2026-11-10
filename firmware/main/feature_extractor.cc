#include "feature_extractor.h"
#include "config.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

namespace {
constexpr int kFftSize = 512;
constexpr float kPi = 3.14159265358979323846f;
float* fft_real = nullptr;
float* fft_imag = nullptr;
float* twiddle_real = nullptr;
float* twiddle_imag = nullptr;
float hamming_table[KWS_FRAME_SAMPLES];
int mel_left[KWS_MEL_BINS];
int mel_center[KWS_MEL_BINS];
int mel_right[KWS_MEL_BINS];
bool tables_ready = false;

constexpr int kFramesPerSlide = KWS_SLIDE_SAMPLES / KWS_HOP_SAMPLES;
// Training frames a window as starts 0, HOP, 2*HOP, ... from the window start, so
// the last frame ends (WINDOW - FRAME) % HOP samples before the window end.
// kCarry is how far before the current slide its first frame starts so that the
// newest frame lands exactly where training puts it.
constexpr int kTail = (KWS_WINDOW_SAMPLES - KWS_FRAME_SAMPLES) % KWS_HOP_SAMPLES;
constexpr int kCarry = (kFramesPerSlide - 1) * KWS_HOP_SAMPLES + KWS_FRAME_SAMPLES + kTail - KWS_SLIDE_SAMPLES;
static_assert(KWS_SLIDE_SAMPLES % KWS_HOP_SAMPLES == 0, "a slide must be a whole number of hops");
static_assert(kCarry >= 0 && kCarry <= KWS_SLIDE_SAMPLES, "frame alignment needs a carry within one slide");
static_assert(1 + (KWS_WINDOW_SAMPLES - KWS_FRAME_SAMPLES) / KWS_HOP_SAMPLES == KWS_FEATURE_FRAMES, "frame count");
static_assert(KWS_FRAME_SAMPLES <= kFftSize, "frame longer than the FFT");
float* frames = nullptr;                 // ring of un-normalised log-mel frames
uint16_t frame_peak[KWS_FEATURE_FRAMES]; // peak |sample| of each frame's hop
int next_frame = 0;                      // ring slot the next frame goes into (= oldest frame)
int16_t carry[kCarry];                   // tail of the previous slide

bool init_tables() {
    if (tables_ready) return true;
    fft_real = static_cast<float*>(heap_caps_malloc(kFftSize * sizeof(float), MALLOC_CAP_8BIT));
    fft_imag = static_cast<float*>(heap_caps_malloc(kFftSize * sizeof(float), MALLOC_CAP_8BIT));
    twiddle_real = static_cast<float*>(heap_caps_malloc(kFftSize / 2 * sizeof(float), MALLOC_CAP_8BIT));
    twiddle_imag = static_cast<float*>(heap_caps_malloc(kFftSize / 2 * sizeof(float), MALLOC_CAP_8BIT));
    if (!fft_real || !fft_imag || !twiddle_real || !twiddle_imag) return false;
    for (int i = 0; i < kFftSize / 2; ++i) {
        const float angle = 2.0f * kPi * i / kFftSize;
        twiddle_real[i] = cosf(angle);
        twiddle_imag[i] = -sinf(angle);
    }
    for (int sample = 0; sample < KWS_FRAME_SAMPLES; ++sample) {
        hamming_table[sample] = 0.54f - 0.46f * cosf(2.0f * kPi * sample / (KWS_FRAME_SAMPLES - 1));
    }
    const float mel_min = hz_to_mel(60.0f), mel_max = hz_to_mel(4800.0f);
    float bounds[KWS_MEL_BINS + 2];
    for (int i = 0; i < KWS_MEL_BINS + 2; ++i) {
        float hz = mel_to_hz(mel_min + (mel_max - mel_min) * i / (KWS_MEL_BINS + 1));
        bounds[i] = floorf((kFftSize + 1) * hz / KWS_SAMPLE_RATE);
    }
    for (int band = 0; band < KWS_MEL_BINS; ++band) {
        // Same widening of degenerate (zero-width) bands as build_mel_filterbank()
        // in scripts/train_jarvis.py, so device features match training features.
        mel_left[band] = static_cast<int>(bounds[band]);
        mel_center[band] = std::max(static_cast<int>(bounds[band + 1]), mel_left[band] + 1);
        mel_right[band] = std::max(static_cast<int>(bounds[band + 2]), mel_center[band] + 1);
    }
    tables_ready = true;
    return true;
}

void fft512() {
    for (int i = 1, j = 0; i < kFftSize; ++i) {
        int bit = kFftSize >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            const float real = fft_real[i];
            fft_real[i] = fft_real[j];
            fft_real[j] = real;
            const float imag = fft_imag[i];
            fft_imag[i] = fft_imag[j];
            fft_imag[j] = imag;
        }
    }
    for (int length = 2; length <= kFftSize; length <<= 1) {
        const int half = length >> 1;
        const int twiddle_step = kFftSize / length;
        for (int start = 0; start < kFftSize; start += length) {
            for (int i = 0; i < half; ++i) {
                const float wr = twiddle_real[i * twiddle_step];
                const float wi = twiddle_imag[i * twiddle_step];
                const int even = start + i;
                const int odd = even + half;
                const float odd_real = fft_real[odd] * wr - fft_imag[odd] * wi;
                const float odd_imag = fft_real[odd] * wi + fft_imag[odd] * wr;
                const float even_real = fft_real[even];
                const float even_imag = fft_imag[even];
                fft_real[even] = even_real + odd_real;
                fft_imag[even] = even_imag + odd_imag;
                fft_real[odd] = even_real - odd_real;
                fft_imag[odd] = even_imag - odd_imag;
            }
        }
    }
}
}

bool kws_features_init() {
    if (!init_tables()) return false;
    if (!frames) {
        frames = static_cast<float*>(heap_caps_malloc(KWS_FEATURE_FRAMES * KWS_MEL_BINS * sizeof(float), MALLOC_CAP_8BIT));
        if (!frames) return false;
        // Same value training gets for zero-padded audio: log(max(0, 1e-10)).
        const float floor_value = logf(1e-10f);
        for (int i = 0; i < KWS_FEATURE_FRAMES * KWS_MEL_BINS; ++i) frames[i] = floor_value;
    }
    return true;
}

void kws_features_push(const int16_t* samples) {
    if (!frames) return;
    for (int hop = 0; hop < kFramesPerSlide; ++hop) {
        const int first = hop * KWS_HOP_SAMPLES - kCarry;
        for (int i = 0; i < KWS_FRAME_SAMPLES; ++i) {
            const int index = first + i;
            const int16_t sample = index < 0 ? carry[kCarry + index] : samples[index];
            fft_real[i] = sample * hamming_table[i];
            fft_imag[i] = 0.0f;
        }
        for (int i = KWS_FRAME_SAMPLES; i < kFftSize; ++i) { fft_real[i] = 0.0f; fft_imag[i] = 0.0f; }
        fft512();
        for (int bin = 0; bin <= kFftSize / 2; ++bin) {
            fft_real[bin] = fft_real[bin] * fft_real[bin] + fft_imag[bin] * fft_imag[bin];
        }
        float* row = frames + next_frame * KWS_MEL_BINS;
        for (int band = 0; band < KWS_MEL_BINS; ++band) {
            const int left = mel_left[band], center = mel_center[band], right = mel_right[band];
            float energy = 0.0f;
            for (int bin = left; bin < center && bin <= kFftSize / 2; ++bin) {
                energy += fft_real[bin] * (bin - left) / (center - left);
            }
            for (int bin = center; bin < right && bin <= kFftSize / 2; ++bin) {
                energy += fft_real[bin] * (right - bin) / (right - center);
            }
            row[band] = logf(fmaxf(energy, 1e-10f));
        }
        int32_t peak = 0;
        for (int i = 0; i < KWS_HOP_SAMPLES; ++i) {
            const int32_t magnitude = std::abs(static_cast<int32_t>(samples[hop * KWS_HOP_SAMPLES + i]));
            if (magnitude > peak) peak = magnitude;
        }
        frame_peak[next_frame] = static_cast<uint16_t>(peak);
        next_frame = (next_frame + 1) % KWS_FEATURE_FRAMES;
    }
    for (int i = 0; i < kCarry; ++i) carry[i] = samples[KWS_SLIDE_SAMPLES - kCarry + i];
}

int32_t kws_features_peak() {
    int32_t peak = 0;
    for (int i = 0; i < KWS_FEATURE_FRAMES; ++i) if (frame_peak[i] > peak) peak = frame_peak[i];
    return peak;
}

void kws_features_quantize(int8_t* destination, float scale, int zero_point) {
    const int count = KWS_FEATURE_FRAMES * KWS_MEL_BINS;
    if (!frames) { for (int i = 0; i < count; ++i) destination[i] = static_cast<int8_t>(zero_point); return; }
    double sum = 0.0, squared = 0.0;
    for (int frame = 0; frame < KWS_FEATURE_FRAMES; ++frame) {
        float frame_sum = 0.0f, frame_squared = 0.0f;
        const float* row = frames + frame * KWS_MEL_BINS;
        for (int band = 0; band < KWS_MEL_BINS; ++band) { frame_sum += row[band]; frame_squared += row[band] * row[band]; }
        sum += frame_sum; squared += frame_squared;
    }
    const float mean = static_cast<float>(sum / count);
    const float deviation = fmaxf(sqrtf(fmaxf(static_cast<float>(squared / count) - mean * mean, 0.0f)), 1e-6f);
    const float factor = 1.0f / (deviation * scale);
    int written = 0;
    for (int n = 0; n < KWS_FEATURE_FRAMES; ++n) {
        const float* row = frames + ((next_frame + n) % KWS_FEATURE_FRAMES) * KWS_MEL_BINS;  // oldest first
        for (int band = 0; band < KWS_MEL_BINS; ++band) {
            const int value = static_cast<int>(lroundf((row[band] - mean) * factor)) + zero_point;
            destination[written++] = static_cast<int8_t>(value < -128 ? -128 : (value > 127 ? 127 : value));
        }
    }
}
