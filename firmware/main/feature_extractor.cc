#include "feature_extractor.h"
#include "config.h"
#include <cmath>

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void kws_extract_features(const int16_t* samples, float* output) {
    float minimum[KWS_MEL_BINS + 2], maximum[KWS_MEL_BINS + 2];
    const float mel_min = hz_to_mel(20.0f), mel_max = hz_to_mel(7600.0f);
    for (int i = 0; i < KWS_MEL_BINS + 2; ++i) {
        float hz = mel_to_hz(mel_min + (mel_max - mel_min) * i / (KWS_MEL_BINS + 1));
        minimum[i] = floorf((512 + 1) * hz / KWS_SAMPLE_RATE);
        maximum[i] = floorf((512 + 1) * hz / KWS_SAMPLE_RATE);
    }
    float mean = 0.0f, squared = 0.0f;
    for (int frame = 0; frame < KWS_FEATURE_FRAMES; ++frame) {
        float power[257] = {};
        for (int bin = 0; bin <= 256; ++bin) {
            float real = 0.0f, imag = 0.0f;
            for (int sample = 0; sample < KWS_FRAME_SAMPLES; ++sample) {
                float angle = -2.0f * static_cast<float>(M_PI) * bin * sample / 512.0f;
                float value = samples[frame * KWS_HOP_SAMPLES + sample] * (0.54f - 0.46f * cosf(2.0f * M_PI * sample / (KWS_FRAME_SAMPLES - 1)));
                real += value * cosf(angle); imag += value * sinf(angle);
            }
            power[bin] = real * real + imag * imag;
        }
        for (int band = 0; band < KWS_MEL_BINS; ++band) {
            int left = static_cast<int>(minimum[band]), center = static_cast<int>(minimum[band + 1]), right = static_cast<int>(maximum[band + 2]);
            float energy = 0.0f;
            for (int bin = left; bin < center; ++bin) energy += power[bin] * (bin - left) / fmaxf(1.0f, center - left);
            for (int bin = center; bin < right; ++bin) energy += power[bin] * (right - bin) / fmaxf(1.0f, right - center);
            float value = logf(fmaxf(energy, 1e-10f));
            output[frame * KWS_MEL_BINS + band] = value;
            mean += value; squared += value * value;
        }
    }
    const int count = KWS_FEATURE_FRAMES * KWS_MEL_BINS;
    mean /= count; float deviation = sqrtf(fmaxf(squared / count - mean * mean, 1e-12f));
    for (int i = 0; i < count; ++i) output[i] = (output[i] - mean) / deviation;
}
