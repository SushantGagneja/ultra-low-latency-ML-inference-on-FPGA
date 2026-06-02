#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_cpu.h"
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
    FPGA_SPI_CLOCK_HZ = 40000000,
    FPGA_PIN_MOSI = 11,
    FPGA_PIN_MISO = 13,
    FPGA_PIN_SCLK = 12,
    FPGA_PIN_CS = 10,
    FPGA_PIN_DONE = 9,
};

typedef struct {
    uint32_t start_cycles;
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
            // esp_cpu_get_cycle_count() returns the CCOUNT register (240MHz).
            // 1 cycle = 4.166 ns.
            uint32_t end_cycles = esp_cpu_get_cycle_count();
            uint32_t cycle_delta = end_cycles - ctx.start_cycles;
            int64_t latency_ns = (int64_t)cycle_delta * 1000 / 240; // Convert to ns

            if (err == ESP_OK) {
                ESP_LOGI(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"decision\":%u,\"latency_ns\":%" PRId64
                         ",\"price\":%.2f,\"status\":\"SUCCESS\",\"note\":\"T2T_hardware_latency\"}",
                         esp_timer_get_time(),
                         (unsigned)decision,
                         latency_ns,
                         ctx.price);
            } else {
                ESP_LOGW(TAG,
                         "{\"type\":\"bnn_inference\",\"timestamp_us\":%" PRId64
                         ",\"decision\":3,\"latency_ns\":0,\"status\":\"%s\"}",
                         esp_timer_get_time(),
                         esp_err_to_name(err));
            }
        }
    }
}

void app_main(void)
{
    // Initialize NVS (required for WiFi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

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

        ctx.start_cycles = esp_cpu_get_cycle_count();
        ctx.price = tick.bid_price; // Approximate for logging

        // Push context to result task BEFORE initiating hardware DMA.
        if (xQueueSend(ctx_queue, &ctx, 0) == pdTRUE) {
            // Initiate Zero-Copy DMA SPI TX (Non-blocking) sending raw tick
            // Format to Q format used by the tick parser
            uint32_t bid_p = (uint32_t)(tick.bid_price * 32768.0f);
            uint32_t ask_p = (uint32_t)(tick.ask_price * 32768.0f);
            uint32_t bid_q = (uint32_t)(tick.bid_qty * 65536.0f);
            uint32_t ask_q = (uint32_t)(tick.ask_qty * 65536.0f);
            
            err = bnn_spi_tx_tick(&fpga, bid_p, bid_q, ask_p, ask_q);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to queue SPI DMA TX: %s", esp_err_to_name(err));
                inference_context_t discard;
                xQueueReceive(ctx_queue, &discard, 0);
            }
        } else {
            ESP_LOGW(TAG, "Context queue full, dropping inference request");
        }
    }
}
