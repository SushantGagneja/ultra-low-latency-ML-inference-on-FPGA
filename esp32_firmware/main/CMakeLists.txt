idf_component_register(
    SRCS
        "main.c"
        "quantization.c"
        "spi_interface.c"
        "temporal_features.c"
        "binance_ws.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_timer esp_netif esp_wifi esp_websocket_client json nvs_flash
)
