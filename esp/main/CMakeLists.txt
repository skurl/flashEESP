idf_component_register(SRCS "main.c" "../../flashee.c"
                       INCLUDE_DIRS "../.."
                       REQUIRES esp_partition esp_timer esp_driver_uart
                       PRIV_REQUIRES espressif__esp-nn)
