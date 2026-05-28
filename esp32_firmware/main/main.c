#include <inttypes.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "nvs_flash.h"

#include "quantization.h"
#include "spi_interface.h"
#include "temporal_features.h"
#include "binance_ws.h"

static const char *TAG = "bnn_phase2";

enum {
    FPGA_SPI_CLOCK_HZ = 80000000,
    FPGA_PIN_MOSI = 11,
    FPGA_PIN_MISO = 13,
    FPGA_PIN_SCLK = 12,
    FPGA_PIN_CS = 10,
    FPGA_PIN_DONE = 9,
};


void app_main(void)
{
    bnn_feature_state_t feature_state;
    bnn_quantizer_t quantizer;
    bnn_spi_t fpga;

    // Initialize NVS (required for WiFi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    bnn_feature_state_init(&feature_state);
    bnn_quantizer_init(&quantizer);

    const bnn_spi_config_t spi_cfg = {
        .host = SPI2_HOST,
        .mosi_io = FPGA_PIN_MOSI,
        .miso_io = FPGA_PIN_MISO,
        .sclk_io = FPGA_PIN_SCLK,
        .cs_io = FPGA_PIN_CS,
        .done_io = FPGA_PIN_DONE,
        .clock_hz = FPGA_SPI_CLOCK_HZ,
    };

    esp_err_t err = bnn_spi_init(&fpga, &spi_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "FPGA SPI init failed: %s", esp_err_to_name(err));
        return;
    }

    QueueHandle_t tick_queue = xQueueCreate(128, sizeof(bnn_market_tick_t));
    if (!tick_queue) {
        ESP_LOGE(TAG, "Failed to create tick queue");
        return;
    }

    ESP_LOGI(TAG, "Initializing WiFi...");
    if (bnn_wifi_init_sta() == ESP_OK) {
        ESP_LOGI(TAG, "Starting Binance WebSocket...");
        err = bnn_binance_ws_start(tick_queue);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Binance WebSocket failed to start: %s", esp_err_to_name(err));
            return;
        }
    } else {
        ESP_LOGE(TAG, "WiFi connection failed! Halting.");
        return;
    }

    ESP_LOGI(TAG, "Phase 2 temporal engine online - waiting for market ticks");

    for (;;) {
        bnn_indicators_t indicators;
        bnn_market_tick_t tick;

        // Block indefinitely until a tick arrives from the WebSocket task
        if (xQueueReceive(tick_queue, &tick, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (!bnn_feature_update(&feature_state, &tick, &indicators)) {
            continue;
        }

        const uint16_t spike = bnn_quantize_bipolar(&quantizer, &indicators);
        bnn_decision_t decision = BNN_DECISION_INVALID;
        int64_t latency_ns = 0;

        err = bnn_spi_infer(&fpga, spike, 0u, &decision, &latency_ns);
        if (err == ESP_OK) {
            ESP_LOGI(TAG,
                     "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                     ",\"spike\":\"0x%04x\",\"decision\":%u,\"latency_ns\":%" PRId64
                     ",\"rsi\":%.2f,\"momentum\":%.6f,\"volatility\":%.6f,\"status\":\"SUCCESS\"}",
                     esp_timer_get_time(),
                     spike,
                     (unsigned)decision,
                     latency_ns,
                     indicators.rsi,
                     indicators.momentum,
                     indicators.volatility);
        } else {
            ESP_LOGW(TAG,
                     "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                     ",\"spike\":\"0x%04x\",\"decision\":3,\"latency_ns\":0,\"status\":\"%s\"}",
                     esp_timer_get_time(),
                     spike,
                     esp_err_to_name(err));
        }
    }
}
