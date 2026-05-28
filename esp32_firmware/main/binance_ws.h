#ifndef BNN_BINANCE_WS_H
#define BNN_BINANCE_WS_H

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_err.h"

#include "temporal_features.h"

#define BNN_WIFI_SSID      CONFIG_BNN_WIFI_SSID
#define BNN_WIFI_PASS      CONFIG_BNN_WIFI_PASSWORD
#define BNN_BINANCE_URI    CONFIG_BNN_BINANCE_URI

/**
 * @brief Initialize WiFi and connect to the specified access point.
 * 
 * @return esp_err_t ESP_OK on success
 */
esp_err_t bnn_wifi_init_sta(void);

/**
 * @brief Start the Binance WebSocket client.
 * 
 * @param tick_queue Queue handle where parsed bnn_market_tick_t will be pushed.
 * @return esp_err_t ESP_OK on success
 */
esp_err_t bnn_binance_ws_start(QueueHandle_t tick_queue);

#endif // BNN_BINANCE_WS_H
