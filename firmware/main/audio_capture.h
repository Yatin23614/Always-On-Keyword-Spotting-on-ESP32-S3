#pragma once
#include <cstddef>
#include <cstdint>
#include "esp_err.h"
esp_err_t kws_audio_init();
esp_err_t kws_audio_read(int16_t* destination, size_t samples, size_t* samples_read);
// Diagnostics: times the ADC driver's pool overflowed (audio lost), and samples delivered.
uint32_t kws_audio_overflows();
uint32_t kws_audio_samples();
