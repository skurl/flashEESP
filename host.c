/* ./flashee_host model.bin SEQUENCE [--per-residue]   -> one row of 320 floats per line */
#include "flashee.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin SEQ [--per-residue]\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    void *blob = malloc(len);
    if (fread(blob, 1, len, f) != (size_t)len) return 1;
    flashee_t m;
    if (flashee_open(&m, blob, len)) { fprintf(stderr, "bad model file\n"); return 1; }

    int per = argc > 3 && !strcmp(argv[3], "--per-residue"), n = (int)strlen(argv[2]);
    float *out = malloc((size_t)(per ? n : 1) * m.d * 4);
    clock_t t0 = clock();
    int r = flashee_embed(&m, argv[2], per ? out : NULL, per ? NULL : out);
    fprintf(stderr, "%d residues, %.1f ms\n", r, 1e3 * (clock() - t0) / CLOCKS_PER_SEC);
    if (r < 0) return 1;
    for (int l = 0; l < (per ? n : 1); l++) {
        for (int c = 0; c < m.d; c++) printf("%.6g%c", out[l * m.d + c], c == m.d - 1 ? '\n' : ' ');
    }
    return 0;
}
