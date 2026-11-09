#include "feature_extractor.h"
#include "config.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include <cmath>

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
        bounds[i] = floorf((512 + 1) * hz / KWS_SAMPLE_RATE);
    }
    for (int band = 0; band < KWS_MEL_BINS; ++band) {
        mel_left[band] = static_cast<int>(bounds[band]);
        mel_center[band] = static_cast<int>(bounds[band + 1]);
        mel_right[band] = static_cast<int>(bounds[band + 2]);
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

void kws_extract_features(const int16_t* samples, float* output) {
    if (!init_tables()) {
        for (int i = 0; i < KWS_FEATURE_FRAMES * KWS_MEL_BINS; ++i) output[i] = 0.0f;
        return;
    }
    float mean = 0.0f, squared = 0.0f;
    for (int frame = 0; frame < KWS_FEATURE_FRAMES; ++frame) {
        for (int i = 0; i < kFftSize; ++i) { fft_real[i] = 0.0f; fft_imag[i] = 0.0f; }
        for (int sample = 0; sample < KWS_FRAME_SAMPLES; ++sample) {
            fft_real[sample] = samples[frame * KWS_HOP_SAMPLES + sample] * hamming_table[sample];
        }
        fft512();
        for (int band = 0; band < KWS_MEL_BINS; ++band) {
            const int left = mel_left[band], center = mel_center[band], right = mel_right[band];
            float energy = 0.0f;
            for (int bin = left; bin < center; ++bin) {
                const float power = fft_real[bin] * fft_real[bin] + fft_imag[bin] * fft_imag[bin];
                energy += power * (bin - left) / fmaxf(1.0f, center - left);
            }
            for (int bin = center; bin < right; ++bin) {
                const float power = fft_real[bin] * fft_real[bin] + fft_imag[bin] * fft_imag[bin];
                energy += power * (right - bin) / fmaxf(1.0f, right - center);
            }
            float value = logf(fmaxf(energy, 1e-10f));
            output[frame * KWS_MEL_BINS + band] = value;
            mean += value; squared += value * value;
        }
        taskYIELD();
    }
    const int count = KWS_FEATURE_FRAMES * KWS_MEL_BINS;
    mean /= count; float deviation = sqrtf(fmaxf(squared / count - mean * mean, 1e-12f));
    for (int i = 0; i < count; ++i) output[i] = (output[i] - mean) / deviation;
}
