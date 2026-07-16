/*
 * Nerve -- Example 11: Pong AI (Neuroevolution)
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
 * A neural network controls the left paddle; a rule-based bot controls the
 * right. The network learns to beat the bot through neuroevolution.
 * Population members each play N_MATCHES against the bot; the best are
 * selected and mutated to form the next generation.
 *
 * Architecture : 5 inputs -> 8 hidden -> 1 output
 * Inputs       : ball_x, ball_y, ball_vx, ball_vy, paddle_y (normalised)
 * Output       : > 0.5 = move up, <= 0.5 = move down
 *
 * Build : gcc -O2 11_pong_ai.c -o pong_ai -lm
 * Run   : ./pong_ai
 */

/* Ask the C library for the POSIX declarations this example needs (nanosleep,
 * struct timespec). Under a strict -std=c99 glibc exposes ISO C only, so the
 * declaration would otherwise be missing and the call would be an implicit
 * one -- a hard error on Clang 16+ and under -Werror on GCC. This must sit
 * above every #include, hence its position at the top of the file. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 199309L
#endif

#define NERVE_IMPLEMENTATION
#include "../nerve.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#  include <windows.h>
#  define SLEEP_MS(ms) Sleep(ms)
static void enable_vt100(void) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  m; GetConsoleMode(h, &m);
    SetConsoleMode(h, m | 0x0004);
}
#else
#  include <unistd.h>
/* usleep() was removed from POSIX.1-2008; nanosleep() is its replacement. */
static void SLEEP_MS(long ms)
{
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000);
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}
static void enable_vt100(void) {}
#endif

/* ---- court dimensions ------------------------------------------------- */
#define CW       70
#define CH       22
#define PAD_H     5
#define PAD_XL    2
#define PAD_XR   (CW - 3)
#define BALL_SPD  0.55f
#define PAD_SPD   0.4f
#define BOT_SPD   0.35f

/* ---- evolution params -------------------------------------------------- */
#define POP      30
#define ELITE     4
#define MATCHES   8
#define MAX_TICKS 600
#define N_IN      5
#define N_HID     8
#define N_OUT     1
#define FRAME_MS  16

/* ---- ANSI ------------------------------------------------------------- */
#define CLR()  fputs("\033[H",           stdout)   /* home only — no flash */
#define HIDE() fputs("\033[?25l\033[2J\033[H",stdout) /* hide + one-time clear */
#define SHOW() fputs("\033[?25h",   stdout)
#define GRN  "\033[32m"
#define YEL  "\033[33m"
#define CYN  "\033[36m"
#define RED  "\033[31m"
#define BLD  "\033[1m"
#define DIM  "\033[2m"
#define RST  "\033[0m"

/* ---- game state -------------------------------------------------------- */
typedef struct {
    float bx, by;      /* ball position */
    float vx, vy;      /* ball velocity */
    float ly, ry;      /* paddle top-edge y (float) */
    int   ai_pts, bot_pts;
} Pong;

static void pong_reset(Pong *g)
{
    g->bx = CW * 0.5f;
    g->by = CH * 0.5f;
    g->vx = ((int)nerve_rand_below(2) == 0 ? 1.0f : -1.0f) * BALL_SPD;
    g->vy = (nerve_rand_float() - 0.5f) * BALL_SPD * 1.4f;
    g->ly = (CH - PAD_H) * 0.5f;
    g->ry = (CH - PAD_H) * 0.5f;
}

/* Returns: 0=ongoing, 1=ai scored, -1=bot scored */
static int pong_tick(Pong *g, float ai_out, float bot_out)
{
    float pad_ctr;

    /* move AI paddle */
    if (ai_out > 0.5f) g->ly -= PAD_SPD;
    else               g->ly += PAD_SPD;
    if (g->ly < 0.0f)             g->ly = 0.0f;
    if (g->ly > (float)(CH-PAD_H)) g->ly = (float)(CH-PAD_H);

    /* move bot paddle (tracks ball with limited speed) */
    pad_ctr = g->ry + PAD_H * 0.5f;
    if (g->by < pad_ctr - 0.5f) g->ry -= BOT_SPD;
    else if (g->by > pad_ctr + 0.5f) g->ry += BOT_SPD;
    (void)bot_out;
    if (g->ry < 0.0f)              g->ry = 0.0f;
    if (g->ry > (float)(CH-PAD_H)) g->ry = (float)(CH-PAD_H);

    /* move ball */
    g->bx += g->vx;
    g->by += g->vy;

    /* top / bottom wall */
    if (g->by <= 0.0f)          { g->by =  0.0f; g->vy = -g->vy; }
    if (g->by >= (float)(CH-1)) { g->by = (float)(CH-1); g->vy = -g->vy; }

    /* left paddle collision */
    if (g->bx <= (float)(PAD_XL+1) &&
        g->by >= g->ly && g->by <= g->ly + PAD_H) {
        g->bx = (float)(PAD_XL+1);
        g->vx = fabsf(g->vx) * 1.05f;
        g->vy += ((g->by - (g->ly + PAD_H*0.5f)) / (PAD_H*0.5f)) * 0.15f;
    }

    /* right paddle collision */
    if (g->bx >= (float)(PAD_XR-1) &&
        g->by >= g->ry && g->by <= g->ry + PAD_H) {
        g->bx = (float)(PAD_XR-1);
        g->vx = -fabsf(g->vx) * 1.05f;
        g->vy += ((g->by - (g->ry + PAD_H*0.5f)) / (PAD_H*0.5f)) * 0.15f;
    }

    /* speed cap */
    if (g->vx >  2.0f) g->vx =  2.0f;
    if (g->vx < -2.0f) g->vx = -2.0f;

    /* scoring */
    if (g->bx <= 0.0f) { g->bot_pts++; return -1; }
    if (g->bx >= (float)(CW-1)) { g->ai_pts++; return  1; }
    return 0;
}

/* ---- get inputs ------------------------------------------------------- */
static void get_inputs(const Pong *g, float *inp)
{
    inp[0] = g->bx / (float)CW;
    inp[1] = g->by / (float)CH;
    inp[2] = g->vx;
    inp[3] = g->vy;
    inp[4] = (g->ly + PAD_H * 0.5f) / (float)CH;
}

/* ---- draw ------------------------------------------------------------- */
static void draw(const Pong *g, int gen, int best_ai, float best_fit, int live)
{
    char row[CW + 1];
    int r, c, pi;
    float out_val;

    CLR();
    printf(BLD "  Nerve Pong AI" RST "  generation %d  |  %s\033[K\n\n",
           gen, live ? GRN "LIVE SHOWCASE" RST : DIM "evaluating..." RST);

    /* top border */
    printf("  " CYN "+");
    for (c = 0; c < CW; c++) printf("-");
    printf("+" RST "\n");

    for (r = 0; r < CH; r++) {
        memset(row, ' ', CW);
        row[CW] = '\0';

        /* paddles */
        pi = (int)g->ly;
        if (r >= pi && r <= pi + PAD_H) row[PAD_XL] = row[PAD_XL+1] = '|';
        pi = (int)g->ry;
        if (r >= pi && r <= pi + PAD_H) row[PAD_XR] = row[PAD_XR+1] = '|';

        /* ball */
        if ((int)g->bx >= 0 && (int)g->bx < CW &&
            (int)g->by == r) row[(int)g->bx] = 'o';

        printf("  " CYN "|" RST "%s" CYN "|" RST, row);

        switch (r) {
        case 1:  printf("  AI: " GRN "%d" RST "  Bot: " RED "%d" RST,
                         g->ai_pts, g->bot_pts); break;
        case 3:  printf("  Best AI wins: " YEL "%d" RST, best_ai); break;
        case 5:  printf("  Fitness: " YEL "%.2f" RST, best_fit);   break;
        case 7:  printf("  Net: %d->%d->%d",N_IN,N_HID,N_OUT);     break;
        case 9:  printf("  Pop: %d  Elite: %d", POP, ELITE);        break;
        case 11: printf("  Matches/agent: %d", MATCHES);            break;
        }
        printf("\033[K\n");
    }
    (void)out_val;

    printf("  " CYN "+");
    for (c = 0; c < CW; c++) printf("-");
    printf("+" RST "\n\n");
    printf(DIM "  AI = Left  |  Bot = Right  |  Ctrl+C to exit\033[K\n" RST);
    fflush(stdout);
}

/* ---- evaluate one brain ----------------------------------------------- */
static float evaluate(network_t *brain)
{
    Pong g;
    float inp[N_IN], out;
    int match, tick, res, wins = 0;
    float score = 0.0f;
    int total_rallies = 0;

    for (match = 0; match < MATCHES; match++) {
        pong_reset(&g);
        g.ai_pts = 0; g.bot_pts = 0;
        for (tick = 0; tick < MAX_TICKS; tick++) {
            get_inputs(&g, inp);
            net_compute(brain, inp, &out);
            res = pong_tick(&g, out, 0.0f);
            if (res != 0) {
                total_rallies++;
                if (res == 1) wins++;
                pong_reset(&g);
                g.ai_pts = (res == 1) ? g.ai_pts : g.ai_pts;
            }
        }
        score += (float)g.ai_pts - (float)g.bot_pts * 0.5f;
    }
    return score + (float)wins * 2.0f;
}

/* ---- neuroevolution --------------------------------------------------- */
typedef struct { network_t *brain; float fitness; } Agent;
static Agent agents[POP];

static int cmp_agent(const void *a, const void *b)
{
    const Agent *aa = (const Agent *)a;
    const Agent *ab = (const Agent *)b;
    if (ab->fitness > aa->fitness) return  1;
    if (ab->fitness < aa->fitness) return -1;
    return 0;
}

static void mutate(network_t *net, float rate, float str)
{
    int l, nu, nl;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l-1].no_of_neurons; nl++)
                if (nerve_rand_float() < rate) {
                    float u = nerve_rand_float() + 1e-7f;
                    float v = nerve_rand_float();
                    float z = (float)sqrt(-2.0*log((double)u))
                            * (float)cos(6.28318530718*(double)v);
                    net->layer[l].neuron[nu].weight[nl] += str * z;
                }
}

static void crossover(network_t *c, const network_t *p1, const network_t *p2)
{
    int l, nu, nl;
    for (l = 1; l < c->no_of_layers; l++)
        for (nu = 0; nu < c->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= c->layer[l-1].no_of_neurons; nl++)
                c->layer[l].neuron[nu].weight[nl] =
                    (nerve_rand_float() < 0.5f)
                    ? p1->layer[l].neuron[nu].weight[nl]
                    : p2->layer[l].neuron[nu].weight[nl];
}

/* ---- main ------------------------------------------------------------- */
int main(void)
{
    Pong g;
    float inp[N_IN], out;
    int i, gen, p1, p2, tick, res;
    int best_ai_wins;
    float best_fit;
    int sizes[3];

    sizes[0] = N_IN; sizes[1] = N_HID; sizes[2] = N_OUT;
    nerve_seed((unsigned long)time(NULL));
    enable_vt100();
    HIDE();

    for (i = 0; i < POP; i++) {
        agents[i].brain = net_allocate_l(3, sizes);
        net_set_activation(agents[i].brain, NERVENET_ACTIVATION_TANH);
        net_initialize_xavier(agents[i].brain);
        agents[i].fitness = 0.0f;
    }

    best_ai_wins = 0;
    best_fit     = 0.0f;

    for (gen = 1; ; gen++) {
        /* evaluate */
        for (i = 0; i < POP; i++)
            agents[i].fitness = evaluate(agents[i].brain);

        qsort(agents, POP, sizeof(Agent), cmp_agent);

        best_fit = agents[0].fitness;
        if (agents[0].fitness > best_ai_wins) best_ai_wins = (int)agents[0].fitness;

        /* showcase best agent */
        pong_reset(&g);
        g.ai_pts = 0; g.bot_pts = 0;
        for (tick = 0; tick < MAX_TICKS * 2; tick++) {
            get_inputs(&g, inp);
            net_compute(agents[0].brain, inp, &out);
            res = pong_tick(&g, out, 0.0f);
            draw(&g, gen, best_ai_wins, best_fit, 1);
            SLEEP_MS(FRAME_MS);
            if (res != 0) pong_reset(&g);
        }

        /* evolve */
        for (i = ELITE; i < POP; i++) {
            p1 = (int)nerve_rand_below(ELITE);
            p2 = ((int)nerve_rand_below(ELITE - 1) + p1 + 1) % ELITE;
            crossover(agents[i].brain, agents[p1].brain, agents[p2].brain);
            mutate(agents[i].brain, 0.15f, 0.2f);
        }
        for (i = POP - 3; i < POP; i++)
            net_randomize(agents[i].brain, 1.0f);
    }

    for (i = 0; i < POP; i++) net_free(agents[i].brain);
    SHOW();
    return 0;
}
