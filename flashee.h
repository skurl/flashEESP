/* FlashEE forward pass in plain C99. No dependencies beyond libc + libm.
 * Weights are read in place from the model.bin blob (flash-mapped on the ESP32). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define FLASHEE_MAX_LAYERS 8

typedef struct { const float *scale; const uint8_t *q; int out, in; } nf4_t;

typedef struct {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *b1, *b2;
    nf4_t wq, wk, wv, wo, w1, w2;
} flashee_layer_t;

typedef struct {
    int d, heads, layers, dff, vocab, classes;
    const uint8_t *tok;                  /* 'A'..'Z' -> token id */
    const float *embed, *lnf_w, *lnf_b;
    flashee_layer_t layer[FLASHEE_MAX_LAYERS];
    nf4_t fc;
} flashee_t;

/* Parse the blob. Nothing is copied: blob must outlive m. 0 on success. */
int flashee_open(flashee_t *m, const void *blob, size_t len);

/* Embed one amino-acid string. per_residue: [strlen(seq)][d] or NULL. pooled: [d] or NULL.
 * Returns the residue count, or <0 on error (allocation). */
int flashee_embed(const flashee_t *m, const char *seq, float *per_residue, float *pooled);
