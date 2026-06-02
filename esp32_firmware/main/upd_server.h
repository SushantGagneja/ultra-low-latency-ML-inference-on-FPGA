#ifndef BNN_UDP_SERVER_H
#define BNN_UDP_SERVER_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// Define the raw market tick structure that perfectly matches the Gateway's
// 17-byte UDP payload. We receive raw floats (Little-Endian matches Xtensa
// float layout), plus a 1-byte metadata field.
typedef struct __attribute__((packed)) {
  float bid_price;
  float ask_price;
  float bid_qty;
  float ask_qty;
  uint8_t metadata;
} bnn_market_tick_t;

// Initialize WiFi connection (Station mode)
esp_err_t bnn_wifi_init_sta(void);

// Start the UDP Server on the specified port.
// Received 16-byte payloads will be cast to bnn_market_tick_t and pushed to the
// queue.
esp_err_t bnn_udp_server_start(uint16_t port, QueueHandle_t tick_queue);

#endif // BNN_UDP_SERVER_H
