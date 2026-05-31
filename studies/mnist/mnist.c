/*
 * Nerve — Benchmark: MNIST handwritten-digit recognition
 * Copyright (C) 2022 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  THE "is it real?" TEST.
 *
 *  MNIST is machine learning's universal yardstick: 70 000 handwritten
 *  digits (28x28 grayscale). If a library can't learn MNIST, it can't learn.
 *
 *  This trains a plain fully-connected Nerve network — no convolutions, no
 *  external math libraries, no Python — straight on the raw pixels and
 *  reports test accuracy on the 10 000-image hold-out set.
 *
 *  Architecture: 784-128-10 | Adam + ReLU(He) | softmax + cross-entropy
 *
 *  Data (IDX files) expected in ./data :
 *     train-images-idx3-ubyte   train-labels-idx1-ubyte
 *     t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
 *
 *  Build:  gcc -O2 mnist.c -o mnist -lm
 *  Run:    ./mnist [epochs] [train_cap]      (defaults: 10  60000)
 */

#define NERVE_IMPLEMENTATION
#include "../../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IMG   784      /* 28 x 28 pixels                                  */
#define CLS    10      /* digit classes 0..9                              */
#define HID   128      /* hidden units                                    */

#define TRAIN_IMG "data/train-images-idx3-ubyte"
#define TRAIN_LBL "data/train-labels-idx1-ubyte"
#define TEST_IMG  "data/t10k-images-idx3-ubyte"
#define TEST_LBL  "data/t10k-labels-idx1-ubyte"

/* ── Big-endian 32-bit reader (IDX files are MSB-first) ────────────────── */
static int read_be32(FILE *f)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

/* Load IDX images -> float[n*IMG], pixels scaled to [0,1]. Returns n.     */
static int load_images(const char *path, float **out)
{
    FILE *f = fopen(path, "rb");
    int magic, n, rows, cols, i;
    unsigned char *buf;
    float *data;
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    magic = read_be32(f); n = read_be32(f);
    rows  = read_be32(f); cols = read_be32(f);
    if (magic != 2051 || rows * cols != IMG) {
        fprintf(stderr, "ERROR: bad image file %s\n", path); fclose(f); return -1;
    }
    buf  = (unsigned char *)malloc((size_t)n * IMG);
    data = (float *)malloc((size_t)n * IMG * sizeof(float));
    fread(buf, 1, (size_t)n * IMG, f);
    fclose(f);
    for (i = 0; i < n * IMG; i++) data[i] = buf[i] / 255.0f;
    free(buf);
    *out = data;
    return n;
}

/* Load IDX labels -> one-hot float[n*CLS]; also fills raw[n] if non-NULL. */
static int load_labels(const char *path, float **out, int *raw)
{
    FILE *f = fopen(path, "rb");
    int magic, n, i;
    unsigned char *buf;
    float *data;
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    magic = read_be32(f); n = read_be32(f);
    if (magic != 2049) {
        fprintf(stderr, "ERROR: bad label file %s\n", path); fclose(f); return -1;
    }
    buf  = (unsigned char *)malloc((size_t)n);
    fread(buf, 1, (size_t)n, f);
    fclose(f);
    data = (float *)calloc((size_t)n * CLS, sizeof(float));
    for (i = 0; i < n; i++) {
        data[i * CLS + buf[i]] = 1.0f;
        if (raw) raw[i] = buf[i];
    }
    free(buf);
    *out = data;
    return n;
}

/* ASCII thumbnail of a 28x28 image (for the sample-prediction display).   */
static void show_digit(const float *px)
{
    static const char ramp[] = " .:-=+*#%@";
    int r, c;
    for (r = 0; r < 28; r += 2) {
        printf("    ");
        for (c = 0; c < 28; c += 1) {
            float v = px[r * 28 + c];
            putchar(ramp[(int)(v * 9.0f + 0.5f)]);
        }
        putchar('\n');
    }
}

int main(int argc, char **argv)
{
    int   epochs    = (argc > 1) ? atoi(argv[1]) : 10;
    int   train_cap = (argc > 2) ? atoi(argv[2]) : 60000;
    float *Xtr, *Ytr, *Xte, *Yte;
    int   *lbl_te;
    int   ntr, nte, e;

    printf("Nerve %s -- MNIST handwritten-digit recognition\n", net_get_version());
    printf("Architecture: 784-%d-10 | Adam + ReLU (He init) | softmax + cross-entropy\n\n",
           HID);

    /* 1. Load --------------------------------------------------------------*/
    ntr = load_images(TRAIN_IMG, &Xtr);
    nte = load_images(TEST_IMG,  &Xte);
    if (load_labels(TRAIN_LBL, &Ytr, NULL) != ntr) return 1;
    lbl_te = (int *)malloc((size_t)nte * sizeof(int));
    if (load_labels(TEST_LBL,  &Yte, lbl_te) != nte) return 1;
    if (ntr <= 0 || nte <= 0) return 1;
    if (train_cap < ntr) ntr = train_cap;   /* optional subset for speed     */
    printf("Train: %d images   Test: %d images   (%d epochs)\n\n",
           ntr, nte, epochs);

    /* 2. Build network -----------------------------------------------------*/
    srand(42);
    network_t *net = net_allocate(3, IMG, HID, CLS);
    net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net, NERVENET_ACTIVATION_RELU);
    net_set_classification(net);   /* softmax output + cross-entropy loss */
    net_initialize_he(net);
    net_set_learning_rate(net, 0.001f);
    net_set_l2_lambda(net, 1e-5f);

    /* 3. Train, reporting hold-out accuracy after every epoch -------------*/
    printf("  epoch    train loss   test acc    sec\n");
    printf("  --------------------------------------\n");
    for (e = 1; e <= epochs; e++) {
        clock_t t0 = clock();
        float loss = net_train_epoch(net, Xtr, Ytr, ntr, IMG, CLS, 1);
        float acc = net_compute_accuracy(net, Xte, Yte, nte, IMG, CLS);
        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("  %4d     %.5f     %6.2f%%   %5.1f\n",
               e, loss, 100.0f * acc, secs);
        fflush(stdout);
    }

    /* 4. Final report: per-digit accuracy via confusion matrix ------------*/
    int cm[CLS * CLS];
    memset(cm, 0, sizeof cm);
    net_confusion_matrix(net, Xte, Yte, nte, IMG, CLS, CLS, cm);

    printf("\n=== Per-digit accuracy (test set) ===\n");
    for (int d = 0; d < CLS; d++) {
        int tot = 0, ok = cm[d * CLS + d];
        for (int p = 0; p < CLS; p++) tot += cm[d * CLS + p];
        printf("   digit %d : %5.2f%%  (%d/%d)\n",
               d, tot ? 100.0f * ok / tot : 0.0f, ok, tot);
    }

    /* 5. A few worked examples (with ASCII art) ---------------------------*/
    printf("\n=== Sample predictions ===\n");
    for (int i = 0; i < 3; i++) {
        int idx = i * 1234 + 7;            /* arbitrary spread of test images */
        int pred = net_classify(net, Xte + idx * IMG);
        printf("\n  true = %d   predicted = %d   %s\n",
               lbl_te[idx], pred, pred == lbl_te[idx] ? "[correct]" : "[wrong]");
        show_digit(Xte + idx * IMG);
    }

    net_free(net);
    free(Xtr); free(Ytr); free(Xte); free(Yte); free(lbl_te);
    return 0;
}
