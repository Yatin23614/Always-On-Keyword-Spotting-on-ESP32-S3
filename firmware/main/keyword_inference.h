#pragma once
#include <cstdint>
#include "esp_err.h"
using kws_command_callback_t = void (*)(uint8_t command_id, float confidence);
esp_err_t kws_inference_init();
// Classify the feature extractor's current window. audio_peak gates out silence.
esp_err_t kws_inference_run(int32_t audio_peak, int64_t now_ms, kws_command_callback_t callback);
