/*
 * Nerve — Geotechnical Case Study: Slope Stability
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  Can Nerve do geotechnical engineering?
 *
 *  This study feeds Nerve 10 000 synthetic slope cases and asks it to
 *  predict the Factor of Safety (FS) — the single most important number
 *  in slope-stability analysis (FS < 1.0 = failure, FS >= 1.5 = a safe
 *  design per most national codes).
 *
 *  Inputs (soil + geometry + groundwater + reinforcement):
 *     1. Unit weight            γ   [kN/m³]
 *     2. Cohesion               c   [kPa]
 *     3. Internal friction      φ   [°]
 *     4. Slope angle            β   [°]
 *     5. Slope height           H   [m]
 *     6. Pore-pressure ratio    ru  [-]
 *     7. Reinforcement type     (one-hot: wall / nailing / geosynth / drain)
 *
 *  Two tasks on the SAME trained network:
 *     A) REGRESSION      — predict the continuous FS value (RMSE, MAE, R²)
 *     B) CLASSIFICATION  — decide "safe (FS>=1.5)?" from the predicted FS
 *                          (accuracy, precision, recall, confusion matrix)
 *
 *  Note on Nerve's output layer: it is always sigmoid (range 0..1), so the
 *  FS target is scaled to [0,1] for training and scaled back for reporting.
 *  FS is physically capped at 3.0 here, which sits naturally at sigmoid's
 *  upper saturation — a good fit for this dataset.
 *
 *  Architecture: 10-24-16-1 | Adam + Tanh | Xavier init | L2 = 1e-4
 *
 *  Build:  gcc -O2 slope_stability.c -o slope_stability -lm
 *  Run:    ./slope_stability            (expects dataset/ alongside the exe)
 */

#define NERVE_IMPLEMENTATION
#include "../../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_PATH  "dataset/slope_stability_dataset.csv"

#define N_NUM      6     /* numeric input features                       */
#define N_REINF    4     /* reinforcement categories (one-hot encoded)   */
#define N_IN       (N_NUM + N_REINF)   /* = 10 network inputs            */
#define N_OUT      1
#define MAX_ROWS   20000

#define FS_LO      0.5f  /* dataset FS range -> target normalisation     */
#define FS_HI      3.0f
#define FS_SAFE    1.5f  /* design threshold: FS >= 1.5 == "safe"        */

#define EPOCHS     200
#define TRAIN_FRAC 0.80f

/* ── Raw dataset (loaded from CSV) ─────────────────────────────────────── */
static float raw_num[MAX_ROWS * N_NUM];  /* 6 numeric features per row    */
static int   raw_reinf[MAX_ROWS];        /* reinforcement class 0..3      */
static float raw_fs[MAX_ROWS];           /* Factor of Safety target       */

/* ── CSV loader ────────────────────────────────────────────────────────── */
/* Columns: γ, c, φ, β, H, ru, "Reinforcement Type", reinf_numeric, FS     */
static int load_csv(const char *path)
{
    FILE *f = fopen(path, "r");
    char  line[512];
    int   n = 0;

    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    if (!fgets(line, sizeof line, f)) { fclose(f); return -1; }   /* header */

    while (n < MAX_ROWS && fgets(line, sizeof line, f)) {
        double g, c, phi, beta, h, ru, fs;
        int    reinf;
        char   type[32];
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%31[^,],%d,%lf",
                   &g, &c, &phi, &beta, &h, &ru, type, &reinf, &fs) == 9) {
            float *row = raw_num + n * N_NUM;
            row[0] = (float)g;    row[1] = (float)c;   row[2] = (float)phi;
            row[3] = (float)beta; row[4] = (float)h;   row[5] = (float)ru;
            raw_reinf[n] = (reinf >= 0 && reinf < N_REINF) ? reinf : 0;
            raw_fs[n]    = (float)fs;
            n++;
        }
    }
    fclose(f);
    return n;
}

/* ── Helpers ───────────────────────────────────────────────────────────── */
static float norm(float x, float lo, float hi)
{ return (hi > lo) ? (x - lo) / (hi - lo) : 0.0f; }

static float denorm_fs(float y)
{ return y * (FS_HI - FS_LO) + FS_LO; }

static const char *reinf_name(int r)
{
    switch (r) {
    case 0:  return "Retaining Wall";
    case 1:  return "Soil Nailing";
    case 2:  return "Geosynthetics";
    default: return "Drainage";
    }
}

int main(void)
{
    static float inputs[MAX_ROWS * N_IN];
    static float targets[MAX_ROWS * N_OUT];
    static int   order[MAX_ROWS];

    float  fmin[N_NUM], fmax[N_NUM];
    int    n, n_train, n_test, i, j, e;

    printf("Nerve %s -- Geotechnical Slope-Stability Study\n", net_get_version());
    printf("Architecture: 10-24-16-1 | Adam + Tanh | Xavier init\n\n");

    /* 1. Load -------------------------------------------------------------- */
    n = load_csv(DATA_PATH);
    if (n <= 0) return 1;
    n_train = (int)(n * TRAIN_FRAC);
    n_test  = n - n_train;
    printf("Loaded %d slope cases  (%d train / %d test)\n", n, n_train, n_test);

    /* 2. Shuffle the row order (deterministic seed) ------------------------ */
    nerve_seed(42);
    for (i = 0; i < n; i++) order[i] = i;
    for (i = n - 1; i > 0; i--) {
        j = (int)nerve_rand_below(i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }

    /* 3. Feature scaling bounds — computed on the TRAIN split only --------- */
    for (j = 0; j < N_NUM; j++) { fmin[j] = 1e30f; fmax[j] = -1e30f; }
    for (i = 0; i < n_train; i++) {
        float *row = raw_num + order[i] * N_NUM;
        for (j = 0; j < N_NUM; j++) {
            if (row[j] < fmin[j]) fmin[j] = row[j];
            if (row[j] > fmax[j]) fmax[j] = row[j];
        }
    }

    /* 4. Build normalised input matrix (6 numeric + 4 one-hot) ------------- */
    for (i = 0; i < n; i++) {
        float *src = raw_num + order[i] * N_NUM;
        float *dst = inputs  + i * N_IN;
        for (j = 0; j < N_NUM; j++) dst[j] = norm(src[j], fmin[j], fmax[j]);
        for (j = 0; j < N_REINF; j++) dst[N_NUM + j] = 0.0f;
        dst[N_NUM + raw_reinf[order[i]]] = 1.0f;
        targets[i] = norm(raw_fs[order[i]], FS_LO, FS_HI);
    }

    /* 5. Build & configure the network ------------------------------------ */
    network_t *net = net_allocate(4, N_IN, 24, 16, N_OUT);
    net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net, NERVENET_ACTIVATION_TANH);
    net_initialize_xavier(net);
    net_set_learning_rate(net, 0.003f);
    net_set_l2_lambda(net, 1e-4f);

    /* 6. Train ------------------------------------------------------------- */
    printf("\nTraining for %d epochs...\n", EPOCHS);
    for (e = 1; e <= EPOCHS; e++) {
        float mse = net_train_epoch(net, inputs, targets,
                                    n_train, N_IN, N_OUT, 32);
        if (e % 25 == 0 || e == 1)
            printf("  epoch %4d / %d   train MSE: %.5f\n", e, EPOCHS, mse);
    }

    /* 7A. Regression metrics on the held-out test set --------------------- */
    float ss_res = 0.0f, ss_tot = 0.0f, abs_sum = 0.0f, mean_fs = 0.0f;
    for (i = n_train; i < n; i++) mean_fs += denorm_fs(targets[i]);
    mean_fs /= (float)n_test;

    for (i = n_train; i < n; i++) {
        float out[1];
        net_compute(net, inputs + i * N_IN, out);
        float pred   = denorm_fs(out[0]);
        float actual = denorm_fs(targets[i]);
        float err    = pred - actual;
        ss_res  += err * err;
        ss_tot  += (actual - mean_fs) * (actual - mean_fs);
        abs_sum += fabsf(err);
    }
    float rmse = sqrtf(ss_res / (float)n_test);
    float mae  = abs_sum / (float)n_test;
    float r2   = 1.0f - ss_res / ss_tot;

    printf("\n=== A) FACTOR-OF-SAFETY REGRESSION (test set, n=%d) ===\n", n_test);
    printf("  RMSE : %.4f  FS units\n", rmse);
    printf("  MAE  : %.4f  FS units\n", mae);
    printf("  R^2  : %.4f\n", r2);
    printf("  (baseline RMSE if always predicting the mean FS: %.4f)\n",
           sqrtf(ss_tot / (float)n_test));

    /* 7B. Stability classification derived from the predicted FS ---------- */
    /* Confusion matrix rows = actual, cols = predicted; 1 = safe (FS>=1.5)  */
    int cm[2][2] = {{0, 0}, {0, 0}};
    for (i = n_train; i < n; i++) {
        float out[1];
        net_compute(net, inputs + i * N_IN, out);
        int pred_safe = denorm_fs(out[0])   >= FS_SAFE;
        int true_safe = denorm_fs(targets[i]) >= FS_SAFE;
        cm[true_safe][pred_safe]++;
    }
    int tp = cm[1][1], tn = cm[0][0], fp = cm[0][1], fn = cm[1][0];
    float acc  = (float)(tp + tn) / (float)n_test;
    float prec = (tp + fp) ? (float)tp / (tp + fp) : 0.0f;
    float rec  = (tp + fn) ? (float)tp / (tp + fn) : 0.0f;

    printf("\n=== B) SAFE-SLOPE CLASSIFICATION  (threshold FS >= %.1f) ===\n",
           FS_SAFE);
    printf("  Accuracy  : %.2f %%\n", 100.0f * acc);
    printf("  Precision : %.2f %%   (of predicted-safe, how many truly safe)\n",
           100.0f * prec);
    printf("  Recall    : %.2f %%   (of truly-safe, how many we caught)\n",
           100.0f * rec);
    printf("\n  Confusion matrix\n");
    printf("                    predicted UNSAFE   predicted SAFE\n");
    printf("    actual UNSAFE   %12d   %14d\n", tn, fp);
    printf("    actual SAFE     %12d   %14d\n", fn, tp);

    /* 8. A few worked examples ------------------------------------------- */
    printf("\n=== Sample predictions ===\n");
    printf("   gamma   c    phi  beta   H    ru  reinforcement | FS act  FS pred\n");
    printf("  -----------------------------------------------------------------\n");
    for (i = n_train; i < n_train + 8 && i < n; i++) {
        float out[1];
        net_compute(net, inputs + i * N_IN, out);
        float *src = raw_num + order[i] * N_NUM;
        printf("  %5.1f %5.1f %5.1f %5.1f %5.1f %4.2f  %-14s| %5.2f   %5.2f\n",
               src[0], src[1], src[2], src[3], src[4], src[5],
               reinf_name(raw_reinf[order[i]]),
               denorm_fs(targets[i]), denorm_fs(out[0]));
    }

    net_free(net);
    return 0;
}
