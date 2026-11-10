#pragma once
#include "driver/gpio.h"
#include "hal/adc_types.h"
#define KWS_SAMPLE_RATE 10000
#define KWS_ADC_SAMPLE_RATE 20000
#define KWS_WINDOW_SAMPLES 28000
#define KWS_FRAME_SAMPLES 300
#define KWS_HOP_SAMPLES 200
#define KWS_MEL_BINS 40
#define KWS_FEATURE_FRAMES 139
#define KWS_CLASS_COUNT 2
#define KWS_CONFIDENCE_THRESHOLD 0.65f
// Longer than the 2.8 s window so one utterance fires once.
#define KWS_COOLDOWN_MS 3000
// Minimum audio peak amplitude to trigger speech evaluation (Voice Activity Detection).
// Room noise floor is ~2800â€“3600; conversational speaking voice reaches 15000â€“32000.
#define KWS_SPEECH_MIN_PEAK 5000
// Sliding window stride for inference (3000 samples = 300 ms @ 10 kHz).
#define KWS_SLIDE_SAMPLES 3000
// ESP32 DevKit V1 analog microphone input.
// GPIO34 is ADC1_CHANNEL_6 and remains usable while Wi-Fi is active.
#define KWS_MIC_ADC_UNIT ADC_UNIT_1
#define KWS_MIC_ADC_CHANNEL ADC_CHANNEL_6
#define KWS_MIC_ADC_GPIO GPIO_NUM_34

// Wi-Fi network for the live command dashboard.
// Change these before sharing the source or firmware publicly.
#if __has_include("wifi_credentials.h")
#include "wifi_credentials.h"
#else
#define KWS_WIFI_SSID "your-ssid"
#define KWS_WIFI_PASSWORD "your-password"
#endif
