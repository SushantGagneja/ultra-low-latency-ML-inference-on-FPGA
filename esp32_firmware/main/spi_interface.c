#include "spi_interface.h"

#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"

esp_err_t bnn_spi_init(bnn_spi_t *iface, const bnn_spi_config_t *cfg)
{
    memset(iface, 0, sizeof(*iface));
    iface->done_io = cfg->done_io;

    // Allocate DMA-capable buffers
    iface->tx_buf = heap_caps_malloc(4, MALLOC_CAP_DMA);
    iface->rx_buf = heap_caps_malloc(4, MALLOC_CAP_DMA);
    if (!iface->tx_buf || !iface->rx_buf) {
        return ESP_ERR_NO_MEM;
    }
    
    memset(&iface->trans_tx, 0, sizeof(spi_transaction_t));
    iface->trans_tx.length = 24;
    iface->trans_tx.tx_buffer = iface->tx_buf;
    iface->trans_tx.rx_buffer = NULL;

    memset(&iface->trans_rx, 0, sizeof(spi_transaction_t));
    iface->trans_rx.length = 8;
    iface->trans_rx.tx_buffer = NULL;
    iface->trans_rx.rx_buffer = iface->rx_buf;

    spi_bus_config_t buscfg = {
        .mosi_io_num = cfg->mosi_io,
        .miso_io_num = cfg->miso_io,
        .sclk_io_num = cfg->sclk_io,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 32,
    };

    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = cfg->clock_hz,
        .mode = 0,
        .spics_io_num = cfg->cs_io,
        .queue_size = 2,
        .command_bits = 0,
        .address_bits = 0,
        .dummy_bits = 0,
    };

    // Use Auto DMA channel for zero-copy transfers
    esp_err_t err = spi_bus_initialize(cfg->host, &buscfg, SPI_DMA_CH_AUTO);
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
            // We enable interrupt on rising edge for DONE signal
            .intr_type = GPIO_INTR_POSEDGE,
        };
        return gpio_config(&io_conf);
    }

    return ESP_OK;
}

esp_err_t bnn_spi_tx_async(bnn_spi_t *iface, uint16_t spike_vector, uint8_t control)
{
    // Pack 24-bit SPI frame: [23:16]=control byte, [15:0]=payload.
    // FIX: Previously masked control to 2 bits ((control & 0x3u) << 16),
    // placing it at bits [17:16]. The FPGA checks packet_sclk[23] to
    // distinguish BRAM writes from inference, so the full byte must
    // occupy [23:16]. For inference: control=0x00. For BRAM write: control=0x80.
    const uint32_t packet = ((uint32_t)control << 16) | spike_vector;
    
    // Write to DMA-capable buffer
    iface->tx_buf[0] = (uint8_t)((packet >> 16) & 0xffu);
    iface->tx_buf[1] = (uint8_t)((packet >> 8) & 0xffu);
    iface->tx_buf[2] = (uint8_t)(packet & 0xffu);

    // Queue transaction (non-blocking)
    return spi_device_queue_trans(iface->dev, &iface->trans_tx, 0);
}

esp_err_t bnn_spi_rx_sync(bnn_spi_t *iface, bnn_decision_t *decision)
{
    spi_transaction_t *ret_trans;
    
    // Get the result of the TX transaction to clear it from the queue
    esp_err_t err = spi_device_get_trans_result(iface->dev, &ret_trans, portMAX_DELAY);
    if (err != ESP_OK) {
        return err;
    }

    // Now issue the synchronous RX transaction to read the decision
    err = spi_device_transmit(iface->dev, &iface->trans_rx);
    if (err != ESP_OK) {
        return err;
    }

    if (decision != NULL) {
        *decision = (bnn_decision_t)(iface->rx_buf[0] & 0x3u);
    }

    return ESP_OK;
}
