#include "spi_interface.h"

#include <string.h>

#include "esp_timer.h"

esp_err_t bnn_spi_init(bnn_spi_t *iface, const bnn_spi_config_t *cfg)
{
    memset(iface, 0, sizeof(*iface));
    iface->done_io = cfg->done_io;

    spi_bus_config_t buscfg = {
        .mosi_io_num = cfg->mosi_io,
        .miso_io_num = cfg->miso_io,
        .sclk_io_num = cfg->sclk_io,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4,
    };

    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = cfg->clock_hz,
        .mode = 0,
        .spics_io_num = cfg->cs_io,
        .queue_size = 1,
        .command_bits = 0,
        .address_bits = 0,
        .dummy_bits = 0,
    };

    esp_err_t err = spi_bus_initialize(cfg->host, &buscfg, SPI_DMA_DISABLED);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        return err;
    }

    err = spi_bus_add_device(cfg->host, &devcfg, &iface->dev);
    if (err != ESP_OK) {
        return err;
    }

    if (cfg->done_io >= 0) {
        gpio_config_t io_conf = {
            .pin_bit_mask = 1ULL << cfg->done_io,
            .mode = GPIO_MODE_INPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_ENABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        return gpio_config(&io_conf);
    }

    return ESP_OK;
}

esp_err_t bnn_spi_infer(bnn_spi_t *iface,
                        uint16_t spike_vector,
                        uint8_t control,
                        bnn_decision_t *decision,
                        int64_t *latency_ns)
{
    const uint32_t packet = ((uint32_t)(control & 0x3u) << 16) | spike_vector;
    uint8_t tx[3] = {
        (uint8_t)((packet >> 16) & 0xffu),
        (uint8_t)((packet >> 8) & 0xffu),
        (uint8_t)(packet & 0xffu),
    };
    uint8_t rx[3] = {0};

    // We send 24 bits. The FPGA needs time to process.
    spi_transaction_t trans_tx = {
        .length = 24,
        .tx_buffer = tx,
        .rx_buffer = NULL,
    };

    const int64_t start = esp_timer_get_time();
    
    // Step 1: Send spike vector and control to FPGA
    esp_err_t err = spi_device_transmit(iface->dev, &trans_tx);
    if (err != ESP_OK) {
        return err;
    }

    // Step 2: Wait for FPGA to assert DONE pin (inference complete)
    if (iface->done_io >= 0) {
        const int64_t timeout_us = start + 1000;
        while (gpio_get_level(iface->done_io) == 0) {
            if (esp_timer_get_time() > timeout_us) {
                return ESP_ERR_TIMEOUT;
            }
        }
    }

    // Step 3: Read back the 2-bit decision (trigger another 8-bit clock cycle)
    uint8_t rx_data = 0;
    spi_transaction_t trans_rx = {
        .length = 8,
        .tx_buffer = NULL,
        .rx_buffer = &rx_data,
    };
    err = spi_device_transmit(iface->dev, &trans_rx);
    if (err != ESP_OK) {
        return err;
    }

    const int64_t end = esp_timer_get_time();
    if (latency_ns != NULL) {
        *latency_ns = (end - start) * 1000;
    }

    if (decision != NULL) {
        *decision = (bnn_decision_t)(rx_data & 0x3u);
    }

    return ESP_OK;
}
