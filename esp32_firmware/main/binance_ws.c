#include "binance_ws.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_websocket_client.h"
#include "cJSON.h"
#include "esp_timer.h"

static const char *TAG = "binance_ws";

/* WiFi event group */
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static QueueHandle_t s_tick_queue = NULL;
static esp_websocket_client_handle_t s_ws_client = NULL;
static int s_wifi_retry_count = 0;

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        if (s_wifi_retry_count < CONFIG_BNN_WIFI_MAX_RETRY) {
            s_wifi_retry_count++;
            esp_wifi_connect();
            ESP_LOGI(TAG, "Retry WiFi connection (%d/%d)",
                     s_wifi_retry_count, CONFIG_BNN_WIFI_MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
        s_wifi_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t bnn_wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = BNN_WIFI_SSID,
            .password = BNN_WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to ap SSID:%s", BNN_WIFI_SSID);
        return ESP_OK;
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s", BNN_WIFI_SSID);
        return ESP_FAIL;
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
        return ESP_FAIL;
    }
}

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    (void)handler_args;
    (void)base;
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_CONNECTED");
        break;
    case WEBSOCKET_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_DISCONNECTED");
        break;
    case WEBSOCKET_EVENT_DATA:
        if (data->op_code == 1 && data->data_len > 0) {
            if (data->payload_offset != 0 || data->payload_len != data->data_len) {
                ESP_LOGW(TAG, "Dropping fragmented websocket payload");
                break;
            }

            // Allocate a null-terminated buffer for cJSON
            char *json_str = malloc(data->data_len + 1);
            if (!json_str) break;
            memcpy(json_str, data->data_ptr, data->data_len);
            json_str[data->data_len] = '\0';

            cJSON *root = cJSON_Parse(json_str);
            if (root) {
                // BookTicker format: {"u":400900217,"s":"BTCUSDT","b":"25.35190000","B":"31.21000000","a":"25.36520000","A":"40.66000000"}
                cJSON *b_obj = cJSON_GetObjectItem(root, "b");
                cJSON *a_obj = cJSON_GetObjectItem(root, "a");
                cJSON *B_obj = cJSON_GetObjectItem(root, "B");
                cJSON *A_obj = cJSON_GetObjectItem(root, "A");

                if (cJSON_IsString(b_obj) && cJSON_IsString(a_obj) &&
                    cJSON_IsString(B_obj) && cJSON_IsString(A_obj)) {
                    const float bid = strtof(b_obj->valuestring, NULL);
                    const float ask = strtof(a_obj->valuestring, NULL);
                    const float bid_qty = strtof(B_obj->valuestring, NULL);
                    const float ask_qty = strtof(A_obj->valuestring, NULL);

                    if (bid <= 0.0f || ask <= 0.0f || ask < bid ||
                        bid_qty < 0.0f || ask_qty < 0.0f) {
                        cJSON_Delete(root);
                        free(json_str);
                        break;
                    }
                    
                    bnn_market_tick_t tick;
                    tick.price = 0.5f * (bid + ask);
                    tick.bid = bid;
                    tick.ask = ask;
                    tick.volume = bid_qty + ask_qty; // Proxy for trade volume
                    tick.timestamp_us = esp_timer_get_time();

                    if (s_tick_queue) {
                        // Push to queue, drop if full (keeps hot-path lock-free and deterministic)
                        static uint32_t dropped_ticks;
                        if (xQueueSend(s_tick_queue, &tick, 0) != pdTRUE) {
                            dropped_ticks++;
                            if ((dropped_ticks & 0xffu) == 0u) {
                                ESP_LOGW(TAG, "Dropped %" PRIu32 " websocket ticks", dropped_ticks);
                            }
                        }
                    }
                }
                cJSON_Delete(root);
            }
            free(json_str);
        }
        break;
    case WEBSOCKET_EVENT_ERROR:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_ERROR");
        break;
    }
}

esp_err_t bnn_binance_ws_start(QueueHandle_t tick_queue)
{
    if (tick_queue == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    s_tick_queue = tick_queue;

    const esp_websocket_client_config_t websocket_cfg = {
        .uri = BNN_BINANCE_URI,
        .task_stack = 8192, // Ensure enough stack for JSON parsing
    };

    s_ws_client = esp_websocket_client_init(&websocket_cfg);
    if (s_ws_client == NULL) {
        return ESP_FAIL;
    }

    esp_err_t err = esp_websocket_register_events(
        s_ws_client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)s_ws_client);
    if (err != ESP_OK) {
        return err;
    }

    return esp_websocket_client_start(s_ws_client);
}
