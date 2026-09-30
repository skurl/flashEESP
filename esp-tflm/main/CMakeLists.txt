idf_component_register(SRCS "main.cc"
                       PRIV_REQUIRES esp_partition esp_timer esp_wifi esp_netif esp_event nvs_flash esp_driver_uart
                       INCLUDE_DIRS "")
