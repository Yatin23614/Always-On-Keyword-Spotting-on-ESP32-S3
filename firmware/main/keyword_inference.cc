#include "keyword_inference.h"
#include "config.h"
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
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
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), nullptr);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return ESP_ERR_NO_MEM;
    input = interpreter->input(0); output = interpreter->output(0);
    return input->type == kTfLiteInt8 && output->type == kTfLiteInt8 ? ESP_OK : ESP_ERR_INVALID_ARG;
}
esp_err_t kws_inference_run(const float* features, int64_t now_ms, kws_command_callback_t callback) {
    if (!interpreter || !input || !output) return ESP_ERR_INVALID_STATE;
    const int feature_count = KWS_FEATURE_FRAMES * KWS_MEL_BINS;
    for (int i = 0; i < feature_count; ++i) {
        const int value = static_cast<int>(std::lround(features[i] / input->params.scale)) + input->params.zero_point;
        input->data.int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, value)));
    }
    if (interpreter->Invoke() != kTfLiteOk) return ESP_FAIL;
    int best = 0;
    for (int i = 1; i < KWS_CLASS_COUNT; ++i) if (output->data.int8[i] > output->data.int8[best]) best = i;
    float confidence = (output->data.int8[best] - output->params.zero_point) * output->params.scale;
    if (best == KWS_CLASS_COUNT - 1 || confidence < KWS_CONFIDENCE_THRESHOLD) { last_command = 255; return ESP_OK; }
    if (best == last_command && now_ms - last_emit_ms < KWS_COOLDOWN_MS) return ESP_OK;
    last_command = static_cast<uint8_t>(best); last_emit_ms = now_ms;
    if (callback) callback(static_cast<uint8_t>(best), confidence);
    return ESP_OK;
}
