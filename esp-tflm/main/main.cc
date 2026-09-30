/* flashee_int8.tflite on TensorFlow Lite Micro (esp-tflite-micro + esp-nn).
 * Boot: map the model partition, allocate tensors (no self-test run: opening the COM port resets the board,
 * so every boot is on a client's clock). Then serve requests over the console (USB cable)
 * and, once WiFi is up, TCP :5000: one line in (sequence), one line out (mean-pooled vector). Client: ../embed.py */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_netif.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "lwip/sockets.h"
#include "freertos/semphr.h"
#include "nvs_flash.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if !__has_include("wifi_secrets.h")
#error "cp main/wifi_secrets.h.example main/wifi_secrets.h and fill in your WiFi (gitignored)"
#endif
#include "wifi_secrets.h"
#define PORT 5000

static const char *VOCAB = "ABCDEFGHIKLMNPQRSTUVWXYZ";   /* checkpoint vocab: ids 5.. in this order; <pad>0 <cls>2 <unk>3 <eos>4 */
/* Arena = all free PSRAM (~8.2 MB of 8 MB used at L=128). Allocated before WiFi starts, so WiFi stays in internal RAM.
 * ponytail: no headroom; if it stops fitting, reconvert with a smaller L or int8 attention. */

static tflite::MicroInterpreter *interp;
static TfLiteTensor *in_tok, *in_mask, *out;

static int token(char c) {
    c &= ~0x20;
    const char *p = (c >= 'A' && c <= 'Z') ? strchr(VOCAB, c) : nullptr;
    return p ? 5 + (int)(p - VOCAB) : 3;
}

static void set_scalar(TfLiteTensor *t, int i, float v) {          /* float or int8 input */
    if (t->type == kTfLiteFloat32) t->data.f[i] = v;
    else t->data.int8[i] = (int8_t)lrintf(v / t->params.scale) + t->params.zero_point;
}
static float get_scalar(const TfLiteTensor *t, int i) {
    return t->type == kTfLiteFloat32 ? t->data.f[i] : (t->data.int8[i] - t->params.zero_point) * t->params.scale;
}

/* Variable length with a fixed-L model: FULLY_CONNECTED (91% of the MACs, and each one streams its weights from
 * flash once per row) only computes the first g_rows = residues + 2 rows; the rest are set to 0.0. That matters:
 * the pad mask is only -16 on the logits (MASK_BIAS in convert.py), so stale values in pad rows leaked into real
 * rows (cos 0.993 -> 0.986); with pad K = V = 0 they contribute ~e^-16. ponytail: attention/elementwise ops still
 * run all L rows; bucketed models if those start to dominate. */
static TFLMRegistration fc_esp_nn;
static int g_rows, g_L;
static TfLiteStatus fc_real_rows(TfLiteContext *ctx, TfLiteNode *node) {
    TfLiteEvalTensor *o = tflite::micro::GetEvalOutput(ctx, node, 0);
    TfLiteIntArray *full = o->dims;
    const int r = full->size - 2;                         /* [.., L, C]: rows dim */
    if (r < 0 || full->data[r] != g_L || o->type != kTfLiteInt8 || tflite::micro::GetTensorShape(o).FlatSize() != g_L * full->data[r + 1])
        return fc_esp_nn.invoke(ctx, node);
    alignas(TfLiteIntArray) static int buf[1 + 8];
    TfLiteIntArray *cut = reinterpret_cast<TfLiteIntArray *>(buf);
    cut->size = full->size;
    for (int i = 0; i < full->size; i++) cut->data[i] = full->data[i];
    cut->data[r] = g_rows;
    o->dims = cut;                                        /* swap the pointer: full->data may live in flash */
    TfLiteStatus st = fc_esp_nn.invoke(ctx, node);
    o->dims = full;
    const int C = full->data[r + 1], zp = static_cast<const tflite::OpDataFullyConnected *>(node->user_data)->output_zero_point;
    memset(o->data.int8 + g_rows * C, zp, (size_t)(g_L - g_rows) * C);
    return st;
}

/* Same idea for attention: convert.py --qchunk splits it into query blocks, each a chain
 * SLICE(q, begin=c) -> RESHAPE -> BATCH_MATMUL -> MUL -> RESHAPE -> ADD -> SOFTMAX -> RESHAPE -> BATCH_MATMUL -> RESHAPE
 * feeding one CONCATENATION. chunk_of[tensor] = the block's first query row (-1 elsewhere); ops whose block starts at
 * or past g_rows are skipped. Their rows only reach the o-projection FC, which skips them too. */
static int16_t *chunk_of;
static TFLMRegistration chunk_orig[6];
template <int K> static TfLiteStatus chunk_skip(TfLiteContext *ctx, TfLiteNode *node) {
    const int c = chunk_of[node->outputs->data[0]];
    if (c >= 0 && c >= g_rows) return kTfLiteOk;
    return chunk_orig[K].invoke(ctx, node);
}
template <int K> static TFLMRegistration chunked(TFLMRegistration r) { chunk_orig[K] = r; r.invoke = chunk_skip<K>; return r; }

static void tag_chunks(const tflite::Model *model, int L) {
    const tflite::SubGraph *g = model->subgraphs()->Get(0);
    chunk_of = (int16_t *)heap_caps_malloc(g->tensors()->size() * sizeof(int16_t), MALLOC_CAP_INTERNAL);
    for (unsigned t = 0; t < g->tensors()->size(); t++) chunk_of[t] = -1;
    int blocks = 0;
    for (const tflite::Operator *op : *g->operators()) {    /* execution order, so one forward pass propagates */
        const auto *code = model->operator_codes()->Get(op->opcode_index());
        const int b = std::max((int)code->builtin_code(), (int)code->deprecated_builtin_code());
        if (b == tflite::BuiltinOperator_CONCATENATION) continue;
        int c = -1;
        if (b == tflite::BuiltinOperator_SLICE) {
            const tflite::Tensor *in = g->tensors()->Get(op->inputs()->Get(0)), *out = g->tensors()->Get(op->outputs()->Get(0));
            const int r = in->shape()->size() - 2;
            if (r >= 0 && in->shape()->Get(r) == L && out->shape()->Get(r) < L) {   /* a query block, not a RoPE half */
                const auto *begin = model->buffers()->Get(g->tensors()->Get(op->inputs()->Get(1))->buffer())->data();
                c = reinterpret_cast<const int32_t *>(begin->data())[r]; blocks++;
            }
        }
        for (int t : *op->inputs()) if (t >= 0 && chunk_of[t] >= 0) c = chunk_of[t];
        if (c >= 0) for (int t : *op->outputs()) chunk_of[t] = c;
    }
    printf("attention: %d query blocks\n", blocks);
}

static SemaphoreHandle_t lock;                            /* serial and TCP share one interpreter */

static void run(const char *seq, int fd) {                /* fd < 0: answer on the console */
    xSemaphoreTake(lock, portMAX_DELAY);
    const int L = in_tok->dims->data[1], d = out->dims->data[2];
    int n = (int)strlen(seq); if (n > L - 2) n = L - 2;
    g_L = L; g_rows = n + 2;
    for (int i = 0; i < L; i++) {
        in_tok->data.i32[i] = i == 0 ? 2 : i == n + 1 ? 4 : i <= n ? token(seq[i - 1]) : 0;
        set_scalar(in_mask, i, i <= n + 1 ? 1.f : 0.f);
    }
    int64_t t0 = esp_timer_get_time();
    if (interp->Invoke() != kTfLiteOk) { printf("Invoke failed\n"); xSemaphoreGive(lock); return; }
    int64_t us = esp_timer_get_time() - t0;
    static float pooled[1024];
    for (int c = 0; c < d; c++) pooled[c] = 0;
    for (int i = 1; i <= n; i++) for (int c = 0; c < d; c++) pooled[c] += get_scalar(out, i * d + c);
    printf("%d residues  %lld ms  (%.1f ms/residue, FC rows %d of L=%d)\n", n, us / 1000, us / 1000.0 / n, g_rows, L);
    fflush(stdout);
    for (int c = 0; c < d; c++) {
        if (fd < 0) printf("%.5f%c", pooled[c] / n, c + 1 < d ? ' ' : '\n');
        else dprintf(fd, "%.5f%c", pooled[c] / n, c + 1 < d ? ' ' : '\n');
    }
    fflush(stdout);
    xSemaphoreGive(lock);
}

static volatile bool got_ip;
static void on_event(void *, esp_event_base_t base, int32_t id, void *data) {
    static int last_reason;
    if (base == IP_EVENT) got_ip = true;
    else if (id == WIFI_EVENT_STA_DISCONNECTED) {
        int r = ((wifi_event_sta_disconnected_t *)data)->reason;   /* 201 no AP (5 GHz-only? out of range?), 15/204 wrong password */
        if (r != last_reason) printf("wifi: can't join \"%s\", reason %d, retrying\n", WIFI_SSID, last_reason = r);
        esp_wifi_connect();
    } else if (id == WIFI_EVENT_STA_START) esp_wifi_connect();
}
static void wifi_start(void) {                            /* non-blocking; tcp_task waits for the IP */
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t ic = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&ic));
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL);
    wifi_config_t wc = {}; strcpy((char *)wc.sta.ssid, WIFI_SSID); strcpy((char *)wc.sta.password, WIFI_PASS);
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);                        /* modem sleep drops ARP: first connect got "No route to host" */
}

static void tcp_task(void *) {
    while (!got_ip) vTaskDelay(pdMS_TO_TICKS(100));
    esp_netif_ip_info_t ip; esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &ip);
    printf("wifi up: " IPSTR " port %d\n", IP2STR(&ip.ip), PORT);
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = INADDR_ANY;
    bind(srv, (sockaddr *)&a, sizeof a); listen(srv, 1);
    static char line[2048];
    for (;;) {                                   /* ponytail: one client at a time */
        int c = accept(srv, NULL, NULL);
        if (c < 0) continue;
        int n;
        while ((n = recv(c, line, sizeof line - 1, 0)) > 0) {
            line[n] = 0; line[strcspn(line, "\r\n")] = 0;
            if (line[0]) run(line, c);
        }
        close(c);
    }
}

extern "C" void app_main(void) {
    lock = xSemaphoreCreateMutex();
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
    if (!part) { printf("no 'model' partition\n"); return; }
    const void *blob; esp_partition_mmap_handle_t h;
    ESP_ERROR_CHECK(esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &blob, &h));
    const tflite::Model *model = tflite::GetModel(blob);
    if (model->version() != TFLITE_SCHEMA_VERSION) { printf("bad model (schema %lu): flash flashee_int8.tflite at 0x210000\n", (unsigned long)model->version()); return; }

    /* exactly the ops convert.py prints for the "fc" recipe */
    static tflite::MicroMutableOpResolver<16> resolver;
    resolver.AddAdd(chunked<0>(tflite::Register_ADD())); resolver.AddBatchMatMul(chunked<1>(tflite::Register_BATCH_MATMUL())); resolver.AddConcatenation(); resolver.AddDequantize();
    resolver.AddEmbeddingLookup();
    fc_esp_nn = tflite::Register_FULLY_CONNECTED();
    TFLMRegistration fc = fc_esp_nn; fc.invoke = fc_real_rows; resolver.AddFullyConnected(fc); resolver.AddLogistic(); resolver.AddMul(chunked<2>(tflite::Register_MUL()));
    resolver.AddQuantize(); resolver.AddReshape(chunked<3>(tflite::Register_RESHAPE())); resolver.AddRsqrt(); resolver.AddSlice(chunked<4>(tflite::Register_SLICE()));
    resolver.AddSoftmax(chunked<5>(tflite::Register_SOFTMAX())); resolver.AddSub(); resolver.AddSum(); resolver.AddTranspose();

    const size_t ARENA = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    uint8_t *arena = (uint8_t *)heap_caps_malloc(ARENA, MALLOC_CAP_SPIRAM);
    if (!arena) { printf("no PSRAM for the arena\n"); return; }
    static tflite::MicroInterpreter it(model, resolver, arena, ARENA);
    interp = &it;
    if (it.AllocateTensors() != kTfLiteOk) { printf("AllocateTensors failed: model needs more than the %u KB of PSRAM\n", (unsigned)(ARENA / 1024)); return; }
    for (size_t i = 0; i < it.inputs_size(); i++) (it.input(i)->type == kTfLiteInt32 ? in_tok : in_mask) = it.input(i);
    out = it.output(0);
    tag_chunks(model, in_tok->dims->data[1]);
    printf("flashee tflm: L=%d, arena used %u KB, free internal %u KB, psram %u KB\n", in_tok->dims->data[1],
           (unsigned)(it.arena_used_bytes() / 1024), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
           (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    wifi_start();
    xTaskCreate(tcp_task, "tcp", 8192, NULL, 5, NULL);

/* Requests over UART0 = the USB-C port marked COM (WCH chip, shows up as /dev/cu.usbmodem*). */
    uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
    static char line[2048];
    printf("ready: send a sequence + newline over the UART port\n"); fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (line[0]) run(line, -1);
    }
}
