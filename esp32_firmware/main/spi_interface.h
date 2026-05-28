#ifndef BNN_SPI_INTERFACE_H
#define BNN_SPI_INTERFACE_H

#include <stdint.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"

typedef enum {
    BNN_DECISION_BUY = 0,
    BNN_DECISION_HOLD = 1,
    BNN_DECISION_SELL = 2,
    BNN_DECISION_INVALID = 3,
} bnn_decision_t;

typedef struct {
    spi_host_device_t host;
    gpio_num_t mosi_io;
    gpio_num_t miso_io;
    gpio_num_t sclk_io;
    gpio_num_t cs_io;
    gpio_num_t done_io;
    int clock_hz;
} bnn_spi_config_t;

typedef struct {
    spi_device_handle_t dev;
    gpio_num_t done_io;
} bnn_spi_t;

esp_err_t bnn_spi_init(bnn_spi_t *iface, const bnn_spi_config_t *cfg);
esp_err_t bnn_spi_infer(bnn_spi_t *iface,
                        uint16_t spike_vector,
                        uint8_t control,
                        bnn_decision_t *decision,
                        int64_t *latency_ns);

#endif
