/*
 * Nerve — Example 05: Regression
 * Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Demonstrates continuous-output regression on the Auto MPG dataset.
 * A 4-8-8-1 network learns to predict fuel efficiency (mpg) from
 * cylinder count, displacement, horsepower, and vehicle weight.
 *
 * Dataset: UCI Auto MPG (40-sample subset, Quinlan 1993)
 * Architecture: 4-8-8-1 | Adam + Tanh | Xavier Init
 *
 * Build:  gcc -O2 05_regression.c -o regression -lm
 * Run:    ./regression
 */

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_TRAIN   32
#define N_TEST     8
#define N_IN       4
#define N_OUT      1
#define EPOCHS  2000

/* Raw data: { cylinders, displacement, horsepower, weight, mpg }
 * Source: UCI Machine Learning Repository — Auto MPG dataset
 * Split is stratified: both training and test span the full mpg range.  */
static const float raw[N_TRAIN + N_TEST][5] = {
    /* training — stratified across mpg range (14 – 38 mpg)            */
    {8, 307, 130, 3504, 18.0f}, {8, 350, 165, 3693, 15.0f},
    {8, 318, 150, 3436, 18.0f}, {8, 304, 150, 3433, 16.0f},
    {8, 429, 198, 4341, 15.0f}, {8, 454, 220, 4354, 14.0f},
    {8, 440, 215, 4312, 14.0f}, {8, 455, 225, 4425, 14.0f},
    {8, 383, 170, 3563, 15.0f}, {8, 340, 160, 3609, 15.0f},
    {8, 400, 150, 3761, 16.0f}, {8, 455, 225, 3086, 14.0f},
    {6, 198,  95, 2833, 22.0f}, {6, 199,  97, 2774, 18.0f},
    {6, 200,  85, 2587, 21.0f}, {4,  97,  88, 2130, 27.0f},
    {4, 110,  87, 2672, 25.0f}, {4, 107,  90, 2430, 24.0f},
    {4, 104,  95, 2375, 25.0f}, {4, 121, 113, 2234, 26.0f},
    {6, 232, 100, 2789, 22.0f}, {6, 250,  88, 3302, 18.0f},
    {6, 250, 100, 3288, 18.0f}, {6, 232, 100, 3275, 16.0f},
    {4, 141,  71, 3190, 19.0f}, {4, 108,  94, 2379, 31.0f},
    {4,  97,  67, 2145, 26.0f}, {4, 105,  63, 2215, 30.0f},
    {4,  91,  53, 1795, 38.0f}, {4,  97,  75, 2265, 33.0f},
    {4, 140,  83, 2640, 27.0f}, {6, 232, 100, 2914, 17.0f},
    /* test — one sample per mpg bracket                                */
    {8, 302, 140, 3449, 17.0f}, {8, 390, 190, 3850, 15.0f},
    {4, 113,  95, 2372, 24.0f}, {6, 199,  90, 2648, 24.0f},
    {4,  97,  46, 1835, 26.0f}, {4,  97,  88, 2130, 28.0f},
    {4,  85,  70, 1990, 36.0f}, {6, 225, 105, 3121, 18.0f}
};

/* Normalization bounds (dataset-wide min / max) */
#define CYL_LO   4.0f
#define CYL_HI   8.0f
#define DIS_LO  68.0f
#define DIS_HI 455.0f
#define HP_LO   46.0f
#define HP_HI  225.0f
#define WT_LO 1613.0f
#define WT_HI 4732.0f
#define MPG_LO   9.0f
#define MPG_HI  47.0f

static float norm(float x, float lo, float hi)
{ return (x - lo) / (hi - lo); }

static float denorm_mpg(float y)
{ return y * (MPG_HI - MPG_LO) + MPG_LO; }

int main(void)
{
    float      inputs[N_TRAIN * N_IN];
    float      targets[N_TRAIN * N_OUT];
    float      test_in[N_TEST * N_IN];
    float      out[1];
    network_t *net;
    float      mse, actual, pred, err, sq_sum;
    int        i;

    printf("Nerve %s -- Fuel Efficiency Regression\n", net_get_version());
    printf("Architecture: 4-8-8-1 | Adam + Tanh | Xavier Init\n");
    printf("Dataset: UCI Auto MPG subset (%d train / %d test)\n\n",
           N_TRAIN, N_TEST);

    for (i = 0; i < N_TRAIN; i++) {
        inputs[i*N_IN+0] = norm(raw[i][0], CYL_LO, CYL_HI);
        inputs[i*N_IN+1] = norm(raw[i][1], DIS_LO, DIS_HI);
        inputs[i*N_IN+2] = norm(raw[i][2], HP_LO,  HP_HI);
        inputs[i*N_IN+3] = norm(raw[i][3], WT_LO,  WT_HI);
        targets[i]       = norm(raw[i][4], MPG_LO, MPG_HI);
    }
    for (i = 0; i < N_TEST; i++) {
        test_in[i*N_IN+0] = norm(raw[N_TRAIN+i][0], CYL_LO, CYL_HI);
        test_in[i*N_IN+1] = norm(raw[N_TRAIN+i][1], DIS_LO, DIS_HI);
        test_in[i*N_IN+2] = norm(raw[N_TRAIN+i][2], HP_LO,  HP_HI);
        test_in[i*N_IN+3] = norm(raw[N_TRAIN+i][3], WT_LO,  WT_HI);
    }

    srand(42);
    net = net_allocate(4, N_IN, 8, 8, N_OUT);
    net_set_optimizer(net,   NERVENET_OPTIMIZER_ADAM);
    net_set_activation(net,  NERVENET_ACTIVATION_TANH);
    net_initialize_xavier(net);
    net_set_learning_rate(net, 0.005f);
    net_set_l2_lambda(net, 1e-4f);

    printf("Training...\n");
    for (i = 1; i <= EPOCHS; i++) {
        mse = net_train_epoch(net, inputs, targets, N_TRAIN, N_IN, N_OUT, 8);
        if (i % 200 == 0)
            printf("  Epoch %4d | MSE: %.4f\n", i, mse);
    }

    printf("\nPredictions on test set:\n");
    printf("  Cyl  Disp   HP   Weight  |  Actual  Predicted   Error\n");
    printf("  ---------------------------------------------------------\n");
    sq_sum = 0.0f;
    for (i = 0; i < N_TEST; i++) {
        net_compute(net, test_in + i*N_IN, out);
        actual = raw[N_TRAIN+i][4];
        pred   = denorm_mpg(out[0]);
        err    = pred - actual;
        sq_sum += err * err;
        printf("  %3.0f  %4.0f  %3.0f  %5.0f   |  %5.1f     %5.1f     %+.1f\n",
               raw[N_TRAIN+i][0], raw[N_TRAIN+i][1],
               raw[N_TRAIN+i][2], raw[N_TRAIN+i][3],
               actual, pred, err);
    }
    printf("\n  Test RMSE: %.2f mpg\n", (float)sqrt((double)(sq_sum / N_TEST)));

    net_free(net);
    return 0;
}
