/*
 * Nerve — Geotechnical Case Study: Sensitivity / "Did it learn the physics?"
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
 *  A high R^2 only proves the network fits the data. The real question is
 *  whether it learned the *mechanics* of slope stability or just memorised
 *  numbers. So we probe the trained network the way an engineer would:
 *
 *    For every input we nudge it across its range (holding the sample's
 *    other features fixed) and measure how FS responds — averaged over the
 *    whole test distribution to avoid the FS=3.0 ceiling masking the slope.
 *
 *  Soil mechanics predicts the SIGN of each effect:
 *     cohesion c        ↑  -> FS ↑   (more shear strength)
 *     friction φ        ↑  -> FS ↑   (more shear strength)
 *     slope angle β     ↑  -> FS ↓   (steeper drives failure)
 *     slope height H    ↑  -> FS ↓   (taller, heavier sliding mass)
 *     pore ratio ru     ↑  -> FS ↓   (water kills effective stress)
 *     unit weight γ     ↑  -> FS ↓   (heavier driving weight; weak effect)
 *
 *  If the network reproduces these signs — and ranks φ/β/c as the heavy
 *  hitters — it captured the physics, not just the table.
 *
 *  Build:  gcc -O2 slope_sensitivity.c -o slope_sensitivity -lm
 *  Run:    ./slope_sensitivity         (run from this folder)
 */

#define NERVE_IMPLEMENTATION
#include "../../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_PATH  "dataset/slope_stability_dataset.csv"

#define N_NUM      6
#define N_REINF    4
#define N_IN       (N_NUM + N_REINF)
#define N_OUT      1
#define MAX_ROWS   20000

#define FS_LO      0.5f
#define FS_HI      3.0f
#define EPOCHS     200
#define TRAIN_FRAC 0.80f

static float raw_num[MAX_ROWS * N_NUM];
static int   raw_reinf[MAX_ROWS];
static float raw_fs[MAX_ROWS];

static const char *FEAT_NAME[N_NUM] = {
    "Unit weight  gamma", "Cohesion     c    ", "Friction     phi  ",
    "Slope angle  beta ", "Slope height H    ", "Pore ratio   ru   "
};
/* Sign the textbook predicts:  +1 raises FS, -1 lowers FS */
static const int FEAT_SIGN[N_NUM] = { -1, +1, +1, -1, -1, -1 };

static int load_csv(const char *path)
{
    FILE *f = fopen(path, "r");
    char  line[512];
    int   n = 0;
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    if (!fgets(line, sizeof line, f)) { fclose(f); return -1; }
    while (n < MAX_ROWS && fgets(line, sizeof line, f)) {
        double g, c, phi, beta, h, ru, fs; int reinf; char type[32];
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%31[^,],%d,%lf",
                   &g, &c, &phi, &beta, &h, &ru, type, &reinf, &fs) == 9) {
            float *row = raw_num + n * N_NUM;
            row[0]=(float)g; row[1]=(float)c; row[2]=(float)phi;
            row[3]=(float)beta; row[4]=(float)h; row[5]=(float)ru;
            raw_reinf[n] = (reinf >= 0 && reinf < N_REINF) ? reinf : 0;
            raw_fs[n] = (float)fs;
            n++;
        }
    }
    fclose(f);
    return n;
}

static float norm(float x, float lo, float hi)
{ return (hi > lo) ? (x - lo) / (hi - lo) : 0.0f; }
static float denorm_fs(float y) { return y * (FS_HI - FS_LO) + FS_LO; }
static const char *reinf_name(int r)
{
    switch (r) { case 0: return "Retaining Wall"; case 1: return "Soil Nailing";
                 case 2: return "Geosynthetics"; default: return "Drainage"; }
}

int main(void)
{
    static float inputs[MAX_ROWS * N_IN];
    static float targets[MAX_ROWS * N_OUT];
    static int   order[MAX_ROWS];
    float fmin[N_NUM], fmax[N_NUM];
    int   n, n_train, n_test, i, j, e;

    printf("Nerve %s -- Does the model know slope-stability physics?\n\n",
           net_get_version());

    n = load_csv(DATA_PATH);
    if (n <= 0) return 1;
    n_train = (int)(n * TRAIN_FRAC);
    n_test  = n - n_train;

    srand(42);
    for (i = 0; i < n; i++) order[i] = i;
    for (i = n - 1; i > 0; i--) { int k = rand()%(i+1);
        int t=order[i]; order[i]=order[k]; order[k]=t; }

    for (j = 0; j < N_NUM; j++) { fmin[j]=1e30f; fmax[j]=-1e30f; }
    for (i = 0; i < n_train; i++) { float *row = raw_num + order[i]*N_NUM;
        for (j = 0; j < N_NUM; j++) {
            if (row[j]<fmin[j]) fmin[j]=row[j];
            if (row[j]>fmax[j]) fmax[j]=row[j]; } }

    for (i = 0; i < n; i++) {
        float *src = raw_num + order[i]*N_NUM;
        float *dst = inputs  + i*N_IN;
        for (j = 0; j < N_NUM; j++) dst[j] = norm(src[j], fmin[j], fmax[j]);
        for (j = 0; j < N_REINF; j++) dst[N_NUM+j] = 0.0f;
        dst[N_NUM + raw_reinf[order[i]]] = 1.0f;
        targets[i] = norm(raw_fs[order[i]], FS_LO, FS_HI);
    }

    network_t *net = net_allocate(4, N_IN, 24, 16, N_OUT);
    net_set_optimizer(net,  NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net, NERVENET_ACTIVATION_TANH);
    net_initialize_xavier(net);
    net_set_learning_rate(net, 0.003f);
    net_set_l2_lambda(net, 1e-4f);

    printf("Training (200 epochs)...\n");
    for (e = 1; e <= EPOCHS; e++)
        net_train_epoch(net, inputs, targets, n_train, N_IN, N_OUT, 32);

    /* ── Sensitivity: average dFS over each feature's full range ───────── */
    /* For every test sample, push feature j to its low/high normalised end
       (others fixed) and record the FS swing; average across the set.       */
    const float D = 0.10f;            /* +/- nudge in normalised space      */
    printf("\n=== INPUT SENSITIVITY  (averaged over %d test cases) ===\n", n_test);
    printf("  Feature              mean dFS   direction   physics?  rank\n");
    printf("  ---------------------------------------------------------------\n");

    float swing[N_NUM];
    for (j = 0; j < N_NUM; j++) {
        double acc = 0.0;
        for (i = n_train; i < n; i++) {
            float xin[N_IN], out[1];
            memcpy(xin, inputs + i*N_IN, sizeof xin);
            float orig = xin[j];
            float hi = orig + D > 1.0f ? 1.0f : orig + D;
            float lo = orig - D < 0.0f ? 0.0f : orig - D;
            if (hi <= lo) continue;
            xin[j] = hi; net_compute(net, xin, out); float fhi = denorm_fs(out[0]);
            xin[j] = lo; net_compute(net, xin, out); float flo = denorm_fs(out[0]);
            acc += (fhi - flo) / (hi - lo);   /* slope per full norm range  */
        }
        swing[j] = (float)(acc / n_test);     /* signed: FS change across range */
    }

    /* rank by magnitude */
    int rank[N_NUM];
    for (j = 0; j < N_NUM; j++) rank[j] = j;
    for (i = 0; i < N_NUM; i++)
        for (j = i+1; j < N_NUM; j++)
            if (fabsf(swing[rank[j]]) > fabsf(swing[rank[i]])) {
                int t = rank[i]; rank[i] = rank[j]; rank[j] = t; }
    int rank_of[N_NUM];
    for (i = 0; i < N_NUM; i++) rank_of[rank[i]] = i + 1;

    for (j = 0; j < N_NUM; j++) {
        int model_sign = swing[j] > 0 ? +1 : -1;
        const char *dir = swing[j] > 0 ? "raises FS" : "lowers FS";
        const char *ok  = (model_sign == FEAT_SIGN[j]) ? "  OK  " : " WRONG";
        printf("  %-18s   %+7.3f   %-9s   %s    #%d\n",
               FEAT_NAME[j], swing[j], dir, ok, rank_of[j]);
    }

    /* ── Reinforcement effect: swap each sample to all 4 types ─────────── */
    printf("\n=== REINFORCEMENT EFFECT  (mean FS, all other inputs held) ===\n");
    float rfs[N_REINF] = {0,0,0,0};
    for (i = n_train; i < n; i++) {
        float xin[N_IN], out[1];
        memcpy(xin, inputs + i*N_IN, sizeof xin);
        for (j = 0; j < N_REINF; j++) xin[N_NUM+j] = 0.0f;
        for (int k = 0; k < N_REINF; k++) {
            xin[N_NUM+k] = 1.0f;
            net_compute(net, xin, out);
            rfs[k] += denorm_fs(out[0]);
            xin[N_NUM+k] = 0.0f;
        }
    }
    for (j = 0; j < N_REINF; j++)
        printf("  %-16s mean FS = %.3f\n", reinf_name(j), rfs[j]/n_test);

    net_free(net);
    return 0;
}
