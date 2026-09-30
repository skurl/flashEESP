/* Boot: map the model partition, embed ubiquitin, print timing. Then a UART REPL: paste a sequence, get a vector. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "flashee.h"

static const char *UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
static flashee_t model;
static float pooled[320];

static void run(const char *seq) {
    int64_t t0 = esp_timer_get_time();
    int n = flashee_embed(&model, seq, NULL, pooled);
    int64_t us = esp_timer_get_time() - t0;
    if (n < 0) { printf("embed failed (%d): out of memory?\n", n); return; }
    printf("%d residues  %lld ms  (%.1f ms/residue)\npooled[0..7] =", n, us / 1000, us / 1000.0 / n);
    for (int i = 0; i < 8; i++) printf(" %.4f", pooled[i]);
    printf("\n");
}

void app_main(void) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) { printf("no 'model' partition: flash model.bin at 0x210000\n"); return; }
    const void *blob; esp_partition_mmap_handle_t h;
    ESP_ERROR_CHECK(esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &blob, &h));
    int rc = flashee_open(&model, blob, part->size);
    if (rc) { printf("bad model blob (%d): flash model.bin at 0x210000\n", rc); return; }
    printf("flashee: %d layers d=%d  free internal %u KB  psram %u KB\n", model.layers, model.d,
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024, heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
    run(UBIQUITIN);

    uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
    static char line[2048];
    printf("> "); fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0]) run(line);
        printf("> "); fflush(stdout);
    }
}
