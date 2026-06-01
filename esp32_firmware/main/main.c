#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "nvs_flash.h"
#include "driver/gpio.h"

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

typedef struct {
    uint16_t spike;
    int64_t start_time;
    bnn_indicators_t indicators;
    float price;
} inference_context_t;

// Global state for tasks
static bnn_spi_t fpga;
static QueueHandle_t tick_queue;
static QueueHandle_t ctx_queue;
static SemaphoreHandle_t fpga_done_sem;

// ISR triggered on FPGA_PIN_DONE rising edge
static void IRAM_ATTR fpga_done_isr_handler(void *arg)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(fpga_done_sem, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

// Task 2: Waits for FPGA completion, reads decision, logs result
static void fpga_result_task(void *arg)
{
    inference_context_t ctx;
    for (;;) {
        // Wait for FPGA DONE interrupt
        if (xSemaphoreTake(fpga_done_sem, portMAX_DELAY) == pdTRUE) {
            
            // Read context that was queued by the ingestion task
            if (xQueueReceive(ctx_queue, &ctx, 0) != pdTRUE) {
                ESP_LOGE(TAG, "FPGA DONE but no context found!");
                continue;
            }

            bnn_decision_t decision = BNN_DECISION_INVALID;
            esp_err_t err = bnn_spi_rx_sync(&fpga, &decision);
            int64_t end_time = esp_timer_get_time();

            // FIX (latency measurement documentation):
            // esp_timer_get_time() returns microseconds.
            // (end_time - ctx.start_time) is the delta in µs.
            // Multiplying by 1000 converts µs → ns, which is correct.
            // HOWEVER: this measures END-TO-END wall-clock latency including:
            //   - SPI TX DMA setup + 24-bit transfer at 80 MHz (~300 ns)
            //   - FPGA BNN inference (~230 ns at 100 MHz, 23 cycles)
            //   - DONE interrupt propagation + FreeRTOS task wake-up
            //   - SPI RX transaction to read decision
            // This is NOT the raw FPGA inference time (which is ~230 ns).
            // Typical values will be in the tens-of-microseconds range.
            int64_t latency_ns = (end_time - ctx.start_time) * 1000;

            if (err == ESP_OK) {
                ESP_LOGI(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"spike\":\"0x%04x\",\"decision\":%u,\"latency_ns\":%" PRId64
                         ",\"rsi\":%.2f,\"momentum\":%.6f,\"volatility\":%.6f,\"price\":%.2f"
                         ",\"status\":\"SUCCESS\",\"note\":\"wall_clock_e2e\"}",
                         esp_timer_get_time(),
                         ctx.spike,
                         (unsigned)decision,
                         latency_ns,
                         ctx.indicators.rsi,
                         ctx.indicators.momentum,
                         ctx.indicators.volatility,
                         ctx.price);
            } else {
                ESP_LOGW(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"spike\":\"0x%04x\",\"decision\":3,\"latency_ns\":0,\"status\":\"%s\"}",
                         esp_timer_get_time(),
                         ctx.spike,
                         esp_err_to_name(err));
            }
        }
    }
}

void app_main(void)
{
    bnn_feature_state_t feature_state;
    bnn_quantizer_t quantizer;

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

    // Install GPIO ISR service and add handler for DONE pin
    gpio_install_isr_service(0);
    gpio_isr_handler_add(FPGA_PIN_DONE, fpga_done_isr_handler, NULL);

    tick_queue = xQueueCreate(128, sizeof(bnn_market_tick_t));
    ctx_queue = xQueueCreate(16, sizeof(inference_context_t));
    fpga_done_sem = xSemaphoreCreateBinary();

    if (!tick_queue || !ctx_queue || !fpga_done_sem) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS primitives");
        return;
    }

    // Spawn the result task on Core 1
    xTaskCreatePinnedToCore(fpga_result_task, "fpga_result", 4096, NULL, 10, NULL, 1);

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

    ESP_LOGI(TAG, "Phase 5 DMA Temporal Engine online - Waiting for market ticks");

    for (;;) {
        bnn_market_tick_t tick;
        inference_context_t ctx;

        // Block indefinitely until a tick arrives
        if (xQueueReceive(tick_queue, &tick, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (!bnn_feature_update(&feature_state, &tick, &ctx.indicators)) {
            continue;
        }

        ctx.spike = bnn_quantize_bipolar(&quantizer, &ctx.indicators);
        ctx.start_time = esp_timer_get_time();
        ctx.price = ctx.indicators.price;

        // Push context to result task BEFORE initiating hardware DMA.
        // The context is consumed by fpga_result_task after DONE fires.
        if (xQueueSend(ctx_queue, &ctx, 0) == pdTRUE) {
            // Initiate Zero-Copy DMA SPI TX (Non-blocking)
            err = bnn_spi_tx_async(&fpga, ctx.spike, 0u);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to queue SPI DMA TX: %s", esp_err_to_name(err));
                // FIX (ctx_queue race): If TX fails, the context is stuck in
                // ctx_queue with no corresponding DONE interrupt coming.
                // The result task would dequeue this stale context on the NEXT
                // unrelated DONE, pairing it with the wrong inference result.
                // Pop it back out to keep the queue consistent.
                inference_context_t discard;
                xQueueReceive(ctx_queue, &discard, 0);
            }
            // Loop immediately without waiting for inference to finish!
            // Achieves concurrent execution of Xtensa Core parsing next tick while FPGA computes.
        } else {
            ESP_LOGW(TAG, "Context queue full, dropping inference request");
        }
    }
}
                continue;
            }

            bnn_decision_t decision = BNN_DECISION_INVALID;
            esp_err_t err = bnn_spi_rx_sync(&fpga, &decision);
            int64_t end_time = esp_timer_get_time();
            int64_t latency_ns = (end_time - ctx.start_time) * 1000;

            if (err == ESP_OK) {
                ESP_LOGI(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"spike\":\"0x%04x\",\"decision\":%u,\"latency_ns\":%" PRId64
                         ",\"rsi\":%.2f,\"momentum\":%.6f,\"volatility\":%.6f,\"price\":%.2f,\"status\":\"SUCCESS\"}",
                         esp_timer_get_time(),
                         ctx.spike,
                         (unsigned)decision,
                         latency_ns,
                         ctx.indicators.rsi,
                         ctx.indicators.momentum,
                         ctx.indicators.volatility,
                         ctx.price);
            } else {
                ESP_LOGW(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"spike\":\"0x%04x\",\"decision\":3,\"latency_ns\":0,\"status\":\"%s\"}",
                         esp_timer_get_time(),
                         ctx.spike,
                         esp_err_to_name(err));
            }
        }
    }
}

void app_main(void)
{
    bnn_feature_state_t feature_state;
    bnn_quantizer_t quantizer;

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

    // Install GPIO ISR service and add handler for DONE pin
    gpio_install_isr_service(0);
    gpio_isr_handler_add(FPGA_PIN_DONE, fpga_done_isr_handler, NULL);

    tick_queue = xQueueCreate(128, sizeof(bnn_market_tick_t));
    ctx_queue = xQueueCreate(16, sizeof(inference_context_t));
    fpga_done_sem = xSemaphoreCreateBinary();

    if (!tick_queue || !ctx_queue || !fpga_done_sem) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS primitives");
        return;
    }

    // Spawn the result task on Core 1
    xTaskCreatePinnedToCore(fpga_result_task, "fpga_result", 4096, NULL, 10, NULL, 1);

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

    ESP_LOGI(TAG, "Phase 5 DMA Temporal Engine online - Waiting for market ticks");

    for (;;) {
        bnn_market_tick_t tick;
        inference_context_t ctx;

        // Block indefinitely until a tick arrives
        if (xQueueReceive(tick_queue, &tick, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (!bnn_feature_update(&feature_state, &tick, &ctx.indicators)) {
            continue;
        }

        ctx.spike = bnn_quantize_bipolar(&quantizer, &ctx.indicators);
        ctx.start_time = esp_timer_get_time();
        ctx.price = ctx.indicators.price;

        // Push context to result task BEFORE initiating hardware DMA
        if (xQueueSend(ctx_queue, &ctx, 0) == pdTRUE) {
            // Initiate Zero-Copy DMA SPI TX (Non-blocking)
            err = bnn_spi_tx_async(&fpga, ctx.spike, 0u);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to queue SPI DMA TX: %s", esp_err_to_name(err));
            }
            // Loop immediately without waiting for inference to finish!
            // Achieves concurrent execution of Xtensa Core parsing next tick while FPGA computes.
        } else {
            ESP_LOGW(TAG, "Context queue full, dropping inference request");
        }
    }
}
