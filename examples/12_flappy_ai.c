/*
 * Nerve -- Example 12: Flappy Bird AI (Neuroevolution)
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
 * A population of 20 birds fly simultaneously through scrolling pipes.
 * Survivors breed the next generation via Gaussian-mutation neuroevolution.
 * All birds are shown live; the number still flying decreases each run until
 * only the fittest brain remains.
 *
 * Architecture : 5 inputs -> 8 hidden -> 1 output
 * Inputs       : bird_y, bird_vy, pipe_dx, gap_top, gap_bot (normalised)
 * Output       : > 0.5 = flap
 *
 * Build : gcc -O2 12_flappy_ai.c -o flappy_ai -lm
 * Run   : ./flappy_ai
 */

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
#  define SLEEP_MS(ms) usleep((unsigned int)((ms)*1000))
static void enable_vt100(void) {}
#endif

/* ---- world dimensions ------------------------------------------------- */
#define DISP_W   72
#define DISP_H   20
#define BIRD_X    8
#define GRAVITY   0.38f
#define FLAP_VY  -2.2f
#define PIPE_W    3
#define PIPE_GAP  7
#define PIPE_PERIOD 22

/* ---- evolution params -------------------------------------------------- */
#define POP      20
#define ELITE     4
#define N_IN      5
#define N_HID     8
#define N_OUT     1
#define FRAME_MS  45

/* ---- ANSI ------------------------------------------------------------- */
#define CLR()  fputs("\033[2J\033[H",stdout)
#define HIDE() fputs("\033[?25l",   stdout)
#define SHOW() fputs("\033[?25h",   stdout)
#define GRN  "\033[32m"
#define YEL  "\033[33m"
#define CYN  "\033[36m"
#define RED  "\033[31m"
#define BLD  "\033[1m"
#define DIM  "\033[2m"
#define RST  "\033[0m"

/* ---- structures ------------------------------------------------------- */
typedef struct {
    float y;
    float vy;
    int   alive;
    int   pipes_cleared;
    float fitness;
    int   ticks;
    network_t *brain;
} Bird;

typedef struct {
    int x;        /* left edge of pipe (display column) */
    int gap_top;  /* first row of the gap */
    int passed;
} Pipe;

#define MAX_PIPES 8

static Bird  birds[POP];
static Pipe  pipes[MAX_PIPES];
static int   n_pipes;
static int   world_tick;

/* ---- pipe management -------------------------------------------------- */
static void add_pipe(void)
{
    int g;
    if (n_pipes >= MAX_PIPES) return;
    g = 2 + rand() % (DISP_H - PIPE_GAP - 4);
    pipes[n_pipes].x       = DISP_W - 1;
    pipes[n_pipes].gap_top = g;
    pipes[n_pipes].passed  = 0;
    n_pipes++;
}

static void reset_world(void)
{
    n_pipes    = 0;
    world_tick = 0;
    add_pipe();
}

static void reset_bird(Bird *b)
{
    b->y              = (float)(DISP_H / 2);
    b->vy             = 0.0f;
    b->alive          = 1;
    b->pipes_cleared  = 0;
    b->fitness        = 0.0f;
    b->ticks          = 0;
}

/* ---- get inputs for one bird ------------------------------------------ */
static void get_inputs(const Bird *b, float *inp)
{
    int k, best = -1;
    float best_dx = (float)DISP_W;
    for (k = 0; k < n_pipes; k++) {
        float dx = (float)(pipes[k].x - BIRD_X);
        if (dx > -PIPE_W && dx < best_dx) { best_dx = dx; best = k; }
    }
    inp[0] = b->y  / (float)DISP_H;
    inp[1] = (b->vy + 5.0f) / 10.0f;
    if (best >= 0) {
        inp[2] = (float)(pipes[best].x - BIRD_X) / (float)DISP_W;
        inp[3] = (float)pipes[best].gap_top / (float)DISP_H;
        inp[4] = (float)(pipes[best].gap_top + PIPE_GAP) / (float)DISP_H;
    } else {
        inp[2] = 1.0f; inp[3] = 0.4f; inp[4] = 0.6f;
    }
}

/* ---- one world tick --------------------------------------------------- */
static void world_step(void)
{
    int i, k, j, brow, alive_any;
    float inp[N_IN], out;

    /* update birds */
    alive_any = 0;
    for (i = 0; i < POP; i++) {
        if (!birds[i].alive) continue;

        get_inputs(&birds[i], inp);
        net_compute(birds[i].brain, inp, &out);
        if (out > 0.5f) birds[i].vy = FLAP_VY;

        birds[i].vy += GRAVITY;
        birds[i].y  += birds[i].vy;
        birds[i].ticks++;

        brow = (int)birds[i].y;
        if (brow <= 0 || brow >= DISP_H - 1) {
            birds[i].alive = 0;
            birds[i].fitness = (float)birds[i].pipes_cleared * 1000.0f
                             + (float)birds[i].ticks;
            continue;
        }

        /* pipe collision */
        for (k = 0; k < n_pipes; k++) {
            if (brow >= 0 && brow < DISP_H &&
                (BIRD_X >= pipes[k].x && BIRD_X < pipes[k].x + PIPE_W)) {
                if (brow < pipes[k].gap_top ||
                    brow >= pipes[k].gap_top + PIPE_GAP) {
                    birds[i].alive = 0;
                    birds[i].fitness = (float)birds[i].pipes_cleared * 1000.0f
                                     + (float)birds[i].ticks;
                }
            }
        }
        if (birds[i].alive) alive_any = 1;
    }
    (void)alive_any;

    /* update pipes */
    for (k = 0; k < n_pipes; k++) {
        pipes[k].x--;
        if (!pipes[k].passed && pipes[k].x + PIPE_W < BIRD_X) {
            pipes[k].passed = 1;
            for (i = 0; i < POP; i++)
                if (birds[i].alive) birds[i].pipes_cleared++;
        }
    }

    /* remove off-screen pipes */
    j = 0;
    for (k = 0; k < n_pipes; k++)
        if (pipes[k].x + PIPE_W > 0) pipes[j++] = pipes[k];
    n_pipes = j;

    /* spawn new pipe */
    world_tick++;
    if (world_tick % PIPE_PERIOD == 0) add_pipe();
}

/* ---- draw ------------------------------------------------------------- */
static void draw_world(int gen, int alive, int best_pipes)
{
    char grid[DISP_H][DISP_W + 1];
    int r, c, k, i;

    for (r = 0; r < DISP_H; r++) {
        memset(grid[r], ' ', DISP_W);
        grid[r][DISP_W] = '\0';
    }

    /* draw pipes */
    for (k = 0; k < n_pipes; k++) {
        for (c = pipes[k].x; c < pipes[k].x + PIPE_W && c < DISP_W && c >= 0; c++) {
            for (r = 0; r < DISP_H; r++) {
                if (r < pipes[k].gap_top || r >= pipes[k].gap_top + PIPE_GAP)
                    grid[r][c] = '|';
            }
        }
    }

    /* draw birds */
    for (i = 0; i < POP; i++) {
        int brow = (int)birds[i].y;
        if (birds[i].alive && brow >= 0 && brow < DISP_H)
            grid[brow][BIRD_X] = 'O';
    }

    CLR();
    printf(BLD "  Nerve Flappy Bird AI" RST
           "  gen %d  |  alive " GRN "%d" RST "/" GRN "%d" RST
           "  |  best " YEL "%d pipes" RST "\n\n",
           gen, alive, POP, best_pipes);

    printf("  " CYN "+");
    for (c = 0; c < DISP_W; c++) printf("-");
    printf("+" RST "\n");

    for (r = 0; r < DISP_H; r++) {
        printf("  " CYN "|" RST);
        for (c = 0; c < DISP_W; c++) {
            char ch = grid[r][c];
            if      (ch == 'O') printf(GRN "O" RST);
            else if (ch == '|') printf(CYN "|" RST);
            else                putchar(' ');
        }
        printf(CYN "|" RST "\n");
    }

    printf("  " CYN "+");
    for (c = 0; c < DISP_W; c++) printf("-");
    printf("+" RST "\n\n");
    printf(DIM "  Net: %d->%d->%d  |  Pop: %d  |  Elite: %d  |  Ctrl+C to exit\n" RST,
           N_IN, N_HID, N_OUT, POP, ELITE);
    fflush(stdout);
}

/* ---- neuroevolution --------------------------------------------------- */
static int cmp_bird(const void *a, const void *b)
{
    const Bird *ba = (const Bird *)a;
    const Bird *bb = (const Bird *)b;
    if (bb->fitness > ba->fitness) return  1;
    if (bb->fitness < ba->fitness) return -1;
    return 0;
}

static void mutate(network_t *net, float rate, float str)
{
    int l, nu, nl;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l-1].no_of_neurons; nl++)
                if ((float)rand()/(float)RAND_MAX < rate) {
                    float u = (float)rand()/(float)RAND_MAX + 1e-7f;
                    float v = (float)rand()/(float)RAND_MAX;
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
                    ((float)rand()/(float)RAND_MAX < 0.5f)
                    ? p1->layer[l].neuron[nu].weight[nl]
                    : p2->layer[l].neuron[nu].weight[nl];
}

/* ---- main ------------------------------------------------------------- */
int main(void)
{
    int i, gen, alive, best_pipes, p1, p2;
    int sizes[3];

    sizes[0] = N_IN; sizes[1] = N_HID; sizes[2] = N_OUT;
    srand((unsigned int)time(NULL));
    enable_vt100();
    HIDE();

    for (i = 0; i < POP; i++) {
        birds[i].brain = net_allocate_l(3, sizes);
        net_set_activation(birds[i].brain, NERVENET_ACTIVATION_TANH);
        net_initialize_xavier(birds[i].brain);
    }

    best_pipes = 0;

    for (gen = 1; ; gen++) {
        reset_world();
        for (i = 0; i < POP; i++) reset_bird(&birds[i]);

        /* run until all birds dead */
        do {
            alive = 0;
            for (i = 0; i < POP; i++) alive += birds[i].alive;

            world_step();

            /* update fitness for living birds */
            for (i = 0; i < POP; i++)
                if (birds[i].alive)
                    birds[i].fitness = (float)birds[i].pipes_cleared * 1000.0f
                                     + (float)birds[i].ticks;

            if (gen > 1 || world_tick > 5) {
                draw_world(gen, alive, best_pipes);
                SLEEP_MS(FRAME_MS);
            }
        } while (alive > 0);

        /* find global best pipes */
        for (i = 0; i < POP; i++)
            if (birds[i].pipes_cleared > best_pipes)
                best_pipes = birds[i].pipes_cleared;

        /* sort and evolve */
        qsort(birds, POP, sizeof(Bird), cmp_bird);

        for (i = ELITE; i < POP; i++) {
            p1 = rand() % ELITE;
            p2 = (rand() % (ELITE - 1) + p1 + 1) % ELITE;
            crossover(birds[i].brain, birds[p1].brain, birds[p2].brain);
            mutate(birds[i].brain, 0.15f, 0.25f);
        }
        for (i = POP - 2; i < POP; i++)
            net_randomize(birds[i].brain, 1.0f);
    }

    for (i = 0; i < POP; i++) net_free(birds[i].brain);
    SHOW();
    return 0;
}
