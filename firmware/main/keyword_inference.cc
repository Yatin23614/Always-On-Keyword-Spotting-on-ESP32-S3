#include "keyword_inference.h"
#include "config.h"
#include "model_data.h"
#include "web_dashboard.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
namespace {
constexpr size_t kTensorArenaBytes = 96 * 1024;
uint8_t tensor_arena[kTensorArenaBytes];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t last_command = 255;
int64_t last_emit_ms = -1000000000;
}
esp_err_t kws_inference_init() {
    const tflite::Model* model = tflite::GetModel(g_keyword_spotter_model);
    if (!model || model->version() != TFLITE_SCHEMA_VERSION) return ESP_ERR_INVALID_VERSION;
    static tflite::MicroMutableOpResolver<7> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk) {
        return ESP_ERR_INVALID_ARG;
    }
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), nullptr);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return ESP_ERR_NO_MEM;
    input = interpreter->input(0); output = interpreter->output(0);
    return input->type == kTfLiteInt8 && output->type == kTfLiteInt8 ? ESP_OK : ESP_ERR_INVALID_ARG;
}
esp_err_t kws_inference_run(const float* features, int16_t audio_peak, int64_t now_ms, kws_command_callback_t callback) {
    if (!interpreter || !input || !output) return ESP_ERR_INVALID_STATE;

    // Silence / Noise floor Gate (Voice Activity Detection):
    // Ambient room noise floor is ~800–3000; real speech produces > 10000.
    // Suppress false activations on silence.
    if (audio_peak < KWS_SPEECH_MIN_PEAK) {
        last_command = 255;
        kws_web_record_diag(255, 0.0f, audio_peak, now_ms);
        static int64_t last_silence_log = 0;
        if (now_ms - last_silence_log >= 1000) {
            ESP_LOGI("kws_diag", "Mic Peak: %5d | Top Pred: 'silence    ' (  0.0%%) | Gate: SILENCE", audio_peak);
            last_silence_log = now_ms;
        }
        return ESP_OK;
    }

    const int feature_count = KWS_FEATURE_FRAMES * KWS_MEL_BINS;
    for (int i = 0; i < feature_count; ++i) {
        const int value = static_cast<int>(std::lround(features[i] / input->params.scale)) + input->params.zero_point;
        input->data.int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, value)));
    }
    if (interpreter->Invoke() != kTfLiteOk) return ESP_FAIL;
    int best = 0;
    for (int i = 1; i < KWS_CLASS_COUNT; ++i) if (output->data.int8[i] > output->data.int8[best]) best = i;
    float confidence = (output->data.int8[best] - output->params.zero_point) * output->params.scale;

    // Stream telemetry to dashboard
    kws_web_record_diag(static_cast<uint8_t>(best), confidence, audio_peak, now_ms);

    // Periodically log diagnostics once per second
    static int64_t last_log_ms = 0;
    if (now_ms - last_log_ms >= 1000) {
        static const char* labels[] = {"hello_jarvis", "unknown"};
        const char* top_name = best < KWS_CLASS_COUNT ? labels[best] : "unknown";
        ESP_LOGI("kws_diag", "Mic Peak: %5d | Top Pred: '%-12s' (%4.1f%%) | Gate: %s",
                 audio_peak, top_name, confidence * 100.0f,
                 (best == 0 && confidence >= KWS_CONFIDENCE_THRESHOLD) ? "TRIGGERED" : "HELD");
        last_log_ms = now_ms;
    }

    if (best != 0 || confidence < KWS_CONFIDENCE_THRESHOLD) { last_command = 255; return ESP_OK; }
    if (best == last_command && now_ms - last_emit_ms < KWS_COOLDOWN_MS) return ESP_OK;
    last_command = static_cast<uint8_t>(best); last_emit_ms = now_ms;
    if (callback) callback(static_cast<uint8_t>(best), confidence);
    return ESP_OK;
}
