#pragma once
#include <cstdint>

// Streaming log-mel front end. Audio is pushed one slide at a time and only the
// new frames are computed; the last KWS_FEATURE_FRAMES frames are kept in a ring,
// so no raw audio window has to be held in RAM.
bool kws_features_init();
// Consume exactly KWS_SLIDE_SAMPLES new samples.
void kws_features_push(const int16_t* samples);
// Peak absolute amplitude over the audio covered by the current feature window.
int32_t kws_features_peak();
// Normalise the current window (zero mean, unit variance, as in training) and
// write it, oldest frame first, as int8 into the model input tensor.
void kws_features_quantize(int8_t* destination, float scale, int zero_point);
