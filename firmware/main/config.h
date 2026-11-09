#pragma once
#include "driver/gpio.h"
#define KWS_SAMPLE_RATE 16000
#define KWS_WINDOW_SAMPLES 16000
#define KWS_FRAME_SAMPLES 320
#define KWS_HOP_SAMPLES 160
#define KWS_MEL_BINS 40
#define KWS_FEATURE_FRAMES 99
#define KWS_CLASS_COUNT 9
#define KWS_CONFIDENCE_THRESHOLD 0.80f
#define KWS_COOLDOWN_MS 800
// Replace these placeholders for the selected ESP32-S3 board.
#define KWS_I2S_BCLK GPIO_NUM_4
#define KWS_I2S_WS GPIO_NUM_5
#define KWS_I2S_DATA GPIO_NUM_6
