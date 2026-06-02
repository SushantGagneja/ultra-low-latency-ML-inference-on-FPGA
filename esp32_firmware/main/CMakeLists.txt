idf_component_register(
    SRCS
        "main.c"
        "spi_interface.c"
        "udp_server.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_timer esp_netif esp_wifi nvs_flash lwip
)
