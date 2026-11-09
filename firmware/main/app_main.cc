#include "audio_capture.h"
#include "config.h"
#include "feature_extractor.h"
#include "keyword_inference.h"
#include "web_dashboard.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>

static const char* TAG = "kws_app";

static void on_command(uint8_t command_id, float confidence) {
    const int64_t now_ms = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "command=%u confidence=%.3f", command_id, confidence);
    kws_web_record_command(command_id, confidence, now_ms);
}

static void kws_worker_task(void* pvParameters) {
    auto* window = static_cast<int16_t*>(heap_caps_calloc(KWS_WINDOW_SAMPLES, sizeof(int16_t), MALLOC_CAP_8BIT));
    auto* features = static_cast<float*>(heap_caps_malloc(KWS_FEATURE_FRAMES * KWS_MEL_BINS * sizeof(float), MALLOC_CAP_8BIT));
    if (!window || !features) {
        ESP_LOGE(TAG, "audio buffers could not be allocated from heap");
        free(window);
        free(features);
        vTaskDelete(nullptr);
        return;
    }

    // Warm up the continuous ADC DMA and let the DC-blocking filter settle for 1 second
    size_t warmup = 0;
    while (warmup < KWS_WINDOW_SAMPLES) {
        size_t chunk = 0;
        kws_audio_read(window, KWS_WINDOW_SAMPLES - warmup, &chunk);
        warmup += chunk;
    }
    std::fill(window, window + KWS_WINDOW_SAMPLES, 0);

    size_t filled = 0;
    while (true) {
        size_t received = 0;
        esp_err_t err = kws_audio_read(window + filled, KWS_WINDOW_SAMPLES - filled, &received);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "audio read error: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        filled += received;
        if (filled == KWS_WINDOW_SAMPLES) {
            int16_t peak = 0;
            for (int i = 0; i < KWS_WINDOW_SAMPLES; ++i) {
                int16_t val = std::abs(window[i]);
                if (val > peak) peak = val;
            }
            kws_web_update_audio(window, KWS_WINDOW_SAMPLES);
            kws_extract_features(window, features);
            kws_inference_run(features, peak, esp_timer_get_time() / 1000, on_command);
            std::move(window + KWS_SLIDE_SAMPLES, window + KWS_WINDOW_SAMPLES, window);
            filled = KWS_WINDOW_SAMPLES - KWS_SLIDE_SAMPLES;

            // Brief yield to ensure IDLE task on Core 1 resets the watchdog
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

extern "C" void app_main() {
    ESP_ERROR_CHECK(kws_web_start());
    ESP_ERROR_CHECK(kws_audio_init());
    esp_err_t model_result = kws_inference_init();
    if (model_result != ESP_OK) {
        ESP_LOGE(TAG, "model init failed: %s; embed a trained int8 model", esp_err_to_name(model_result));
        return;
    }

    // Pin the KWS DSP & inference worker to Core 1 so Core 0 is dedicated to Wi-Fi and web server
    xTaskCreatePinnedToCore(kws_worker_task, "kws_worker", 8192, nullptr, 5, nullptr, 1);
}
