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
#include <cstddef>

static const char* TAG = "kws_app";

static void on_command(uint8_t command_id, float confidence) {
    const int64_t now_ms = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "command=%u confidence=%.3f", command_id, confidence);
    kws_web_record_command(command_id, confidence, now_ms);
}

static void kws_worker_task(void*) {
    // One slide of new audio at a time; the feature extractor keeps the history.
    static int16_t slide[KWS_SLIDE_SAMPLES];

    // Let the ADC DMA and the DC-blocking filter settle before listening.
    for (int i = 0; i < 4; ++i) {
        size_t got = 0;
        kws_audio_read(slide, KWS_SLIDE_SAMPLES, &got);
    }

    size_t filled = 0;
    int cycles = 0;
    while (true) {
        size_t received = 0;
        const esp_err_t err = kws_audio_read(slide + filled, KWS_SLIDE_SAMPLES - filled, &received);
        filled += received;  // keep whatever arrived before an error
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "audio read error: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (filled < KWS_SLIDE_SAMPLES) continue;
        filled = 0;

        const int64_t t0 = esp_timer_get_time();
        kws_features_push(slide);
        const int64_t t1 = esp_timer_get_time();
        kws_inference_run(kws_features_peak(), t1 / 1000, on_command);
        const int64_t t2 = esp_timer_get_time();
        kws_web_update_audio(slide, KWS_SLIDE_SAMPLES);

        if (++cycles % 20 == 0) {
            static int64_t last_us = 0;
            static uint32_t last_samples = 0;
            const uint32_t samples = kws_audio_samples();
            const int rate = last_us ? static_cast<int>((samples - last_samples) * 1000000LL / (t2 - last_us)) : 0;
            last_us = t2; last_samples = samples;
            ESP_LOGI(TAG, "timing: features %d ms, inference %d ms (budget %d ms) | audio %d Hz, ADC overflows %u | free heap %u",
                     static_cast<int>((t1 - t0) / 1000), static_cast<int>((t2 - t1) / 1000),
                     KWS_SLIDE_SAMPLES * 1000 / KWS_SAMPLE_RATE, rate,
                     static_cast<unsigned>(kws_audio_overflows()),
                     static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)));
        }
    }
}

extern "C" void app_main() {
    if (!kws_features_init()) {
        ESP_LOGE(TAG, "feature buffers could not be allocated");
        return;
    }
    // The dashboard is optional: detection keeps running without Wi-Fi.
    const esp_err_t web_result = kws_web_start();
    if (web_result != ESP_OK) ESP_LOGE(TAG, "dashboard disabled: %s", esp_err_to_name(web_result));
    ESP_ERROR_CHECK(kws_audio_init());
    const esp_err_t model_result = kws_inference_init();
    if (model_result != ESP_OK) {
        ESP_LOGE(TAG, "model init failed: %s; embed a trained int8 model", esp_err_to_name(model_result));
        return;
    }
    // DSP + inference on core 1; core 0 handles Wi-Fi and the web server.
    xTaskCreatePinnedToCore(kws_worker_task, "kws_worker", 8192, nullptr, 5, nullptr, 1);
}
