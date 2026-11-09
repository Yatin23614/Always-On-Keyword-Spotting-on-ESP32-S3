#pragma once

#include <cstdint>
#include <cstddef>
#include "esp_err.h"

esp_err_t kws_web_start();
void kws_web_record_command(uint8_t command_id, float confidence, int64_t now_ms);
void kws_web_record_diag(uint8_t top_id, float top_confidence, int16_t audio_peak, int64_t now_ms);
void kws_web_update_audio(const int16_t* samples, size_t count);
