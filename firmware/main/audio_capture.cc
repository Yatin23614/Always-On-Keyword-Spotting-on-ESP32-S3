#include "audio_capture.h"
#include "config.h"
#include "esp_adc/adc_continuous.h"
#include "esp_attr.h"
#include "esp_check.h"
#include <algorithm>
#include <atomic>
namespace {
adc_continuous_handle_t adc_handle = nullptr;
// Q24.8 fixed-point DC tracker with cutoff ~66 Hz (/ 48 at 20 kHz) to
// eliminate integer deadband and reject 50/60 Hz mains hum while
// preserving speech fundamental frequencies (100–300 Hz).
int32_t dc_accum = 2048 << 8;
std::atomic<uint32_t> pool_overflows{0};
std::atomic<uint32_t> samples_out{0};

bool IRAM_ATTR on_pool_overflow(adc_continuous_handle_t, const adc_continuous_evt_data_t*, void*) {
    pool_overflows.fetch_add(1, std::memory_order_relaxed);
    return false;
}
}

uint32_t kws_audio_overflows() { return pool_overflows.load(); }
uint32_t kws_audio_samples() { return samples_out.load(); }

esp_err_t kws_audio_init() {
    adc_continuous_handle_cfg_t handle_config = {};
    // 16 KB ring buffer holds ~400 ms of 20 kHz ADC samples so DMA never
    // drops frames while Core 1 is computing FFTs or neural network inference.
    handle_config.max_store_buf_size = 16384;
    handle_config.conv_frame_size = 256;
    ESP_RETURN_ON_ERROR(adc_continuous_new_handle(&handle_config, &adc_handle), "kws_adc", "new handle");
    adc_digi_pattern_config_t pattern = {};
    pattern.atten = ADC_ATTEN_DB_12;
    pattern.bit_width = ADC_BITWIDTH_12;
    pattern.channel = KWS_MIC_ADC_CHANNEL;
    pattern.unit = KWS_MIC_ADC_UNIT;
    adc_continuous_config_t config = {};
    config.pattern_num = 1;
    config.adc_pattern = &pattern;
    // ESP32 continuous ADC supports 20 kHz as its minimum rate. Downsample
    // 20 kHz to the model's 10 kHz stream in kws_audio_read().
    config.sample_freq_hz = KWS_ADC_SAMPLE_RATE;
    config.conv_mode = ADC_CONV_SINGLE_UNIT_1;
    // The ESP32 uses TYPE1 output by default; the format field is deprecated in IDF 6.1.
    ESP_RETURN_ON_ERROR(adc_continuous_config(adc_handle, &config), "kws_adc", "config");
    adc_continuous_evt_cbs_t callbacks = {};
    callbacks.on_pool_ovf = on_pool_overflow;
    ESP_RETURN_ON_ERROR(adc_continuous_register_event_callbacks(adc_handle, &callbacks, nullptr), "kws_adc", "callbacks");
    return adc_continuous_start(adc_handle);
}

esp_err_t kws_audio_read(int16_t* destination, size_t samples, size_t* samples_read) {
    uint8_t buffer[256];
    size_t produced = 0;
    while (produced < samples) {
        uint32_t bytes_read = 0;
        const esp_err_t err = adc_continuous_read(adc_handle, buffer, sizeof(buffer), &bytes_read, 1000);
        if (err != ESP_OK) {
            // Report the samples already written so the caller does not lose them.
            if (samples_read) *samples_read = produced;
            return err;
        }
        for (size_t offset = 0; offset + sizeof(adc_digi_output_data_t) <= bytes_read; offset += sizeof(adc_digi_output_data_t)) {
            const auto* sample = reinterpret_cast<const adc_digi_output_data_t*>(buffer + offset);
            if (sample->type1.channel != KWS_MIC_ADC_CHANNEL) continue;
            const int32_t raw = sample->type1.data;
            dc_accum += ((raw << 8) - dc_accum) / 48;
            const int32_t dc_val = dc_accum >> 8;
            // 16x digital gain matches the electret microphone swing to the 16-bit range
            const int32_t scaled = (raw - dc_val) * 16;
            const int16_t centered = static_cast<int16_t>(std::max<int32_t>(-32768, std::min<int32_t>(32767, scaled)));
            // Keep every second sample: 20 kHz -> 10 kHz.
            static bool downsample_toggle = false;
            downsample_toggle = !downsample_toggle;
            if (downsample_toggle) {
                destination[produced++] = centered;
            }
            if (produced == samples) break;
        }
    }
    samples_out.fetch_add(produced, std::memory_order_relaxed);
    if (samples_read) *samples_read = produced;
    return ESP_OK;
}
