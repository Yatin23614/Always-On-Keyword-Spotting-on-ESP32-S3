#include "audio_capture.h"
#include "config.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
namespace { i2s_chan_handle_t rx_handle = nullptr; }

esp_err_t kws_audio_init() {
    i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&channel, nullptr, &rx_handle), "kws_audio", "channel");
    i2s_std_config_t config = {};
    config.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(KWS_SAMPLE_RATE);
    config.slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
    config.gpio_cfg = {I2S_GPIO_UNUSED, KWS_I2S_BCLK, KWS_I2S_WS, KWS_I2S_DATA, I2S_GPIO_UNUSED};
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_handle, &config), "kws_audio", "init");
    return i2s_channel_enable(rx_handle);
}

esp_err_t kws_audio_read(int16_t* destination, size_t samples, size_t* samples_read) {
    size_t bytes = 0;
    esp_err_t result = i2s_channel_read(rx_handle, destination, samples * sizeof(int16_t), &bytes, 1000);
    if (samples_read) *samples_read = bytes / sizeof(int16_t);
    return result;
}
