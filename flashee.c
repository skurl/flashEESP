#include "flashee.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

#define LIN_ROWS 64                      /* weight rows dequantised per block: 64*640 B = 40 KB internal RAM */
#define MAX_IN 640

/* NF4 codebook * 127, rounded. Max relative error 0.4% of the row scale. */
static const int8_t NF4_I8[16] = {-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127};

/* ---- blob parsing ------------------------------------------------------- */

static const uint8_t *take(const uint8_t **p, size_t n) { const uint8_t *r = *p; *p += n; return r; }
static const float *f32(const uint8_t **p, int n) { return (const float *)take(p, (size_t)n * 4); }
static nf4_t nf4(const uint8_t **p, int out, int in) {
    nf4_t w = { f32(p, out), NULL, out, in };
    w.q = take(p, (size_t)out * in / 2);
    return w;
}

int flashee_open(flashee_t *m, const void *blob, size_t len) {
    const uint8_t *p = blob, *end = p + len;
    if (len < 60 || memcmp(p, "FLEE", 4)) return -1;
    const uint32_t *h = (const uint32_t *)(p + 4);
    m->d = h[0]; m->heads = h[1]; m->layers = h[2]; m->dff = h[3]; m->vocab = h[4]; m->classes = h[5];
    if (m->layers > FLASHEE_MAX_LAYERS || m->dff > MAX_IN || m->d > MAX_IN || m->d % 16 || m->dff % 16) return -2;
    p += 28;
    m->tok = take(&p, 32);
    m->embed = f32(&p, m->vocab * m->d);
    for (int i = 0; i < m->layers; i++) {
        flashee_layer_t *l = &m->layer[i];
        l->ln1_w = f32(&p, m->d); l->ln1_b = f32(&p, m->d);
        l->wq = nf4(&p, m->d, m->d); l->wk = nf4(&p, m->d, m->d);
        l->wv = nf4(&p, m->d, m->d); l->wo = nf4(&p, m->d, m->d);
        l->ln2_w = f32(&p, m->d); l->ln2_b = f32(&p, m->d);
        l->w1 = nf4(&p, m->dff, m->d); l->b1 = f32(&p, m->dff);
        l->w2 = nf4(&p, m->d, m->dff); l->b2 = f32(&p, m->d);
    }
    m->lnf_w = f32(&p, m->d); m->lnf_b = f32(&p, m->d);
    m->fc = nf4(&p, m->classes, m->d);
    return p <= end ? 0 : -3;            /* trailing bytes allowed: a flash partition is bigger than the model */
}

/* ---- kernels ------------------------------------------------------------- */

#ifdef CONFIG_IDF_TARGET_ESP32S3
/* esp-nn's PIE kernel (ee.vmulas.s8.accx, 16 MACs/instruction). Both pointers 16-byte aligned, n a multiple of 16. */
int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int n);
#define dot_i8 esp_nn_dot_s8_aligned_esp32s3
#else
static int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}
#endif

/* ponytail: one call per (row, token) costs ~40% overhead on a 640-dot; a fused R-rows kernel is the next step. */

/* per-token symmetric int8 (W4A8: the smallest lossless config in the paper) */
static void quant_rows(const float *x, int L, int n, int8_t *xq, float *sx) {
    for (int l = 0; l < L; l++) {
        float amax = 0;
        for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[l * n + i]));
        float s = amax > 0 ? amax / 127.f : 1.f, inv = 1.f / s;
        sx[l] = s;
        for (int i = 0; i < n; i++) xq[l * n + i] = (int8_t)lrintf(x[l * n + i] * inv);
    }
}

/* y[L][out] = x @ W^T + bias.  Weights are dequantised LIN_ROWS at a time into internal RAM
 * so each activation row is streamed out/LIN_ROWS times, not `out` times. */
static void linear(const nf4_t *w, const float *bias, const int8_t *xq, const float *sx, int L, float *y) {
    static int8_t blk[LIN_ROWS][MAX_IN] __attribute__((aligned(16)));  /* ponytail: static => not reentrant; per-core copy if split across cores */
    const int in = w->in, out = w->out;
    for (int o0 = 0; o0 < out; o0 += LIN_ROWS) {
        int R = out - o0 < LIN_ROWS ? out - o0 : LIN_ROWS;
        for (int r = 0; r < R; r++) {
            const uint8_t *q = w->q + (size_t)(o0 + r) * in / 2;
            for (int i = 0; i < in / 2; i++) { blk[r][2 * i] = NF4_I8[q[i] & 15]; blk[r][2 * i + 1] = NF4_I8[q[i] >> 4]; }
        }
        for (int l = 0; l < L; l++) {
            const int8_t *xr = xq + (size_t)l * in;
            float *yr = y + (size_t)l * out + o0;
            for (int r = 0; r < R; r++)
                yr[r] = dot_i8(xr, blk[r], in) * (sx[l] * w->scale[o0 + r] * (1.f / 127.f)) + (bias ? bias[o0 + r] : 0.f);
        }
    }
}

static void layernorm(const float *x, const float *w, const float *b, int L, int n, float *y) {
    for (int l = 0; l < L; l++) {
        const float *xr = x + l * n; float *yr = y + l * n, mean = 0, var = 0;
        for (int i = 0; i < n; i++) mean += xr[i];
        mean /= n;
        for (int i = 0; i < n; i++) var += (xr[i] - mean) * (xr[i] - mean);
        float inv = 1.f / sqrtf(var / n + 1e-5f);
        for (int i = 0; i < n; i++) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

/* rotate q and k in place; the angle depends on (position, dim pair) only, so it is shared by all heads */
static void rope(float *q, float *k, int L, int d, int heads) {
    int hd = d / heads, half = hd / 2;
    for (int l = 0; l < L; l++)
        for (int c = 0; c < half; c++) {
            float ang = (float)l * (float)(1.0 / pow(10000.0, (2.0 * c) / hd)), cs = cosf(ang), sn = sinf(ang);
            for (int h = 0; h < heads; h++) {
                float *v[2] = { q + l * d + h * hd, k + l * d + h * hd };
                for (int t = 0; t < 2; t++) {
                    float a = v[t][c], b = v[t][c + half];
                    v[t][c] = a * cs - b * sn; v[t][c + half] = b * cs + a * sn;
                }
            }
        }
}

/* ponytail: float attention, O(L^2 * d). Fine to L~256; quantise q/k to int8 + dot_i8 beyond that. */
static void attention(const float *q, const float *k, const float *v, int L, int d, int heads, float *s, float *out) {
    int hd = d / heads; float scale = 1.f / sqrtf((float)hd);
    for (int h = 0; h < heads; h++)
        for (int i = 0; i < L; i++) {
            const float *qi = q + i * d + h * hd; float mx = -1e30f, sum = 0, *o = out + i * d + h * hd;
            for (int j = 0; j < L; j++) {
                const float *kj = k + j * d + h * hd; float a = 0;
                for (int c = 0; c < hd; c++) a += qi[c] * kj[c];
                s[j] = a * scale; if (s[j] > mx) mx = s[j];
            }
            for (int j = 0; j < L; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
            for (int c = 0; c < hd; c++) o[c] = 0;
            for (int j = 0; j < L; j++) {
                float p = s[j] / sum; const float *vj = v + j * d + h * hd;
                for (int c = 0; c < hd; c++) o[c] += p * vj[c];
            }
        }
}

static float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }

/* ---- forward -------------------------------------------------------------- */

int flashee_embed(const flashee_t *m, const char *seq, float *per_residue, float *pooled) {
    const int d = m->d, dff = m->dff, n = (int)strlen(seq), L = n + 2;
    float *x = malloc((size_t)L * d * 4), *h = malloc((size_t)L * d * 4), *qkv = malloc((size_t)L * 3 * d * 4);
    int8_t *xq = aligned_alloc(16, (size_t)L * dff); float *sx = malloc(L * 4), *sc = malloc(L * 4);  /* rows 16-aligned for the PIE dot */
    if (!x || !h || !qkv || !xq || !sx || !sc) { free(x); free(h); free(qkv); free(xq); free(sx); free(sc); return -1; }
    float *q = qkv, *k = qkv + (size_t)L * d, *v = qkv + (size_t)2 * L * d, *ff = qkv;

    for (int l = 0; l < L; l++) {                                 /* <cls> seq <eos> */
        int c = l ? seq[l - 1] & ~0x20 : 0, id = l == 0 ? 2 : l == L - 1 ? 4 : (c >= 'A' && c <= 'Z') ? m->tok[c - 'A'] : 3;
        memcpy(x + l * d, m->embed + id * d, d * 4);
    }
    for (int i = 0; i < m->layers; i++) {
        const flashee_layer_t *ly = &m->layer[i];
        layernorm(x, ly->ln1_w, ly->ln1_b, L, d, h);
        quant_rows(h, L, d, xq, sx);
        linear(&ly->wq, NULL, xq, sx, L, q); linear(&ly->wk, NULL, xq, sx, L, k); linear(&ly->wv, NULL, xq, sx, L, v);
        rope(q, k, L, d, m->heads);
        attention(q, k, v, L, d, m->heads, sc, h);
        quant_rows(h, L, d, xq, sx);
        linear(&ly->wo, NULL, xq, sx, L, q);
        for (size_t j = 0; j < (size_t)L * d; j++) x[j] += q[j];

        layernorm(x, ly->ln2_w, ly->ln2_b, L, d, h);
        quant_rows(h, L, d, xq, sx);
        linear(&ly->w1, ly->b1, xq, sx, L, ff);
        for (size_t j = 0; j < (size_t)L * dff; j++) ff[j] = gelu(ff[j]);
        quant_rows(ff, L, dff, xq, sx);
        linear(&ly->w2, ly->b2, xq, sx, L, h);
        for (size_t j = 0; j < (size_t)L * d; j++) x[j] += h[j];
    }
    layernorm(x, m->lnf_w, m->lnf_b, L, d, h);
    if (per_residue) memcpy(per_residue, h + d, (size_t)n * d * 4);
    if (pooled) {
        for (int c = 0; c < d; c++) pooled[c] = 0;
        for (int l = 1; l <= n; l++) for (int c = 0; c < d; c++) pooled[c] += h[l * d + c];
        for (int c = 0; c < d; c++) pooled[c] /= n;
    }
    free(x); free(h); free(qkv); free(xq); free(sx); free(sc);
    return n;
}
