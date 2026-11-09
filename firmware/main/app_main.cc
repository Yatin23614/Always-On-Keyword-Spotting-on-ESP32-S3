#include "audio_capture.h"
#include "feature_extractor.h"
#include "keyword_inference.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <array>
#include <algorithm>
static const char* TAG = "kws_app";
static void on_command(uint8_t command_id, float confidence) { ESP_LOGI(TAG, "command=%u confidence=%.3f", command_id, confidence); }
extern "C" void app_main() {
    ESP_ERROR_CHECK(kws_audio_init());
    esp_err_t model_result = kws_inference_init();
    if (model_result != ESP_OK) { ESP_LOGE(TAG, "model init failed: %s; embed a trained int8 model", esp_err_to_name(model_result)); return; }
    std::array<int16_t, 16000> window{};
    std::array<float, 99 * 40> features{};
    size_t filled = 0;
    while (true) {
        size_t received = 0;
        ESP_ERROR_CHECK(kws_audio_read(window.data() + filled, window.size() - filled, &received));
        filled += received;
        if (filled == window.size()) {
            kws_extract_features(window.data(), features.data());
            kws_inference_run(features.data(), esp_timer_get_time() / 1000, on_command);
            std::move(window.begin() + 160, window.end(), window.begin());
            filled = window.size() - 160;
        }
    }
}
