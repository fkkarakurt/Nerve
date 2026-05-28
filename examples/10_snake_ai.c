/*
 * Nerve -- Example 10: Snake AI (Neuroevolution)
 * Copyright (c) 2022 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * A population of 100 neural networks learns to play Snake through
 * neuroevolution (tournament selection + Gaussian mutation).
 * After each generation the best brain is showcased live in the terminal.
 *
 * Architecture : 11 inputs -> 16 hidden -> 3 outputs
 * Inputs       : danger (3) | direction one-hot (4) | food direction (4)
 * Outputs      : turn-left / straight / turn-right  (argmax)
 *
 * Build : gcc -O2 10_snake_ai.c -o snake_ai -lm
 * Run   : ./snake_ai
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

/* ---- constants -------------------------------------------------------- */
#define BW       28
#define BH       18
#define POP      100
#define ELITE      5
#define N_IN      11
#define N_HID     16
#define N_OUT      3
#define MAXSTEPS  (BW * BH * 4)
#define FRAME_MS  60

#define UP    0
#define RIGHT 1
#define DOWN  2
#define LEFT  3

static const int DX[4] = { 0, 1, 0,-1};
static const int DY[4] = {-1, 0, 1, 0};

/* ---- ANSI helpers ------------------------------------------------------ */
#define CLR()    fputs("\033[2J\033[H",stdout)
#define HIDE()   fputs("\033[?25l",   stdout)
#define SHOW()   fputs("\033[?25h",   stdout)
#define GRN      "\033[32m"
#define BGRN     "\033[92m"
#define RED      "\033[31m"
#define YEL      "\033[33m"
#define CYN      "\033[36m"
#define DIM      "\033[2m"
#define BLD      "\033[1m"
#define RST      "\033[0m"

/* ---- snake struct ------------------------------------------------------ */
typedef struct {
    int  bx[BW * BH];
    int  by[BW * BH];
    int  len;
    int  dir;
    int  fx, fy;
    int  alive;
    int  score;
    int  steps;
    int  starve;
    float fitness;
    network_t *brain;
} Snake;

static Snake pop[POP];

/* ---- board helpers ----------------------------------------------------- */
static int is_body(const Snake *s, int x, int y)
{
    int i;
    for (i = 1; i < s->len; i++)
        if (s->bx[i] == x && s->by[i] == y) return 1;
    return 0;
}

static void place_food(Snake *s)
{
    int x, y;
    do { x = rand() % BW; y = rand() % BH; }
    while ((x == s->bx[0] && y == s->by[0]) || is_body(s,x,y));
    s->fx = x; s->fy = y;
}

static void reset_snake(Snake *s)
{
    s->bx[0] = BW/2; s->by[0] = BH/2;
    s->bx[1] = BW/2; s->by[1] = BH/2+1;
    s->bx[2] = BW/2; s->by[2] = BH/2+2;
    s->len   = 3; s->dir = UP;
    s->alive = 1; s->score = 0;
    s->steps = 0; s->starve = 0;
    s->fitness = 0.0f;
    place_food(s);
}

static int is_danger(const Snake *s, int x, int y)
{
    return (x < 0 || x >= BW || y < 0 || y >= BH || is_body(s,x,y));
}

/* ---- neural inputs ---------------------------------------------------- */
static void get_inputs(const Snake *s, float *inp)
{
    int hx = s->bx[0], hy = s->by[0];
    int d  = s->dir;
    int dr = (d + 1) % 4;
    int dl = (d + 3) % 4;

    inp[0] = is_danger(s, hx+DX[d],  hy+DY[d])  ? 1.0f : 0.0f;
    inp[1] = is_danger(s, hx+DX[dr], hy+DY[dr]) ? 1.0f : 0.0f;
    inp[2] = is_danger(s, hx+DX[dl], hy+DY[dl]) ? 1.0f : 0.0f;

    inp[3] = (d == UP)    ? 1.0f : 0.0f;
    inp[4] = (d == RIGHT) ? 1.0f : 0.0f;
    inp[5] = (d == DOWN)  ? 1.0f : 0.0f;
    inp[6] = (d == LEFT)  ? 1.0f : 0.0f;

    inp[7]  = (s->fy < hy) ? 1.0f : 0.0f;
    inp[8]  = (s->fx > hx) ? 1.0f : 0.0f;
    inp[9]  = (s->fy > hy) ? 1.0f : 0.0f;
    inp[10] = (s->fx < hx) ? 1.0f : 0.0f;
}

/* ---- one game step ---------------------------------------------------- */
static void step_snake(Snake *s)
{
    float inp[N_IN], out[N_OUT];
    int dec, nd, nx, ny, i;

    if (!s->alive) return;
    get_inputs(s, inp);
    net_compute(s->brain, inp, out);

    dec = 1;
    if (out[0] > out[dec]) dec = 0;
    if (out[2] > out[dec]) dec = 2;

    if      (dec == 0) nd = (s->dir + 3) % 4;
    else if (dec == 2) nd = (s->dir + 1) % 4;
    else               nd = s->dir;
    s->dir = nd;

    nx = s->bx[0] + DX[nd];
    ny = s->by[0] + DY[nd];

    if (nx < 0 || nx >= BW || ny < 0 || ny >= BH || is_body(s,nx,ny)) {
        s->alive = 0; return;
    }

    s->steps++; s->starve++;
    if (s->starve > BW * BH) { s->alive = 0; return; }

    for (i = s->len - 1; i > 0; i--) {
        s->bx[i] = s->bx[i-1];
        s->by[i] = s->by[i-1];
    }
    s->bx[0] = nx; s->by[0] = ny;

    if (nx == s->fx && ny == s->fy) {
        s->score++;
        s->starve = 0;
        if (s->len < BW * BH) s->len++;
        place_food(s);
    }
}

/* ---- fitness ---------------------------------------------------------- */
static void compute_fitness(Snake *s)
{
    s->fitness = (float)s->steps + (float)(s->score * s->score) * 500.0f;
}

/* ---- draw ------------------------------------------------------------- */
static void draw(const Snake *s, int gen, int best_all, float best_fit)
{
    char grid[BH][BW];
    int r, c, i;
    memset(grid, ' ', sizeof(grid));
    grid[s->fy][s->fx] = '*';
    for (i = 1; i < s->len; i++) grid[s->by[i]][s->bx[i]] = '#';
    grid[s->by[0]][s->bx[0]] = '@';

    CLR();
    printf(BLD "  Nerve Snake AI" RST "  generation %d\n\n", gen);

    printf("  " CYN "+");
    for (c = 0; c < BW; c++) printf("--");
    printf("+" RST "\n");

    for (r = 0; r < BH; r++) {
        printf("  " CYN "|" RST);
        for (c = 0; c < BW; c++) {
            char ch = grid[r][c];
            if      (ch == '@') printf(BLD BGRN "@@" RST);
            else if (ch == '#') printf(GRN "##" RST);
            else if (ch == '*') printf(RED "<>" RST);
            else                printf("  ");
        }
        printf(CYN "|" RST);
        switch (r) {
        case 1: printf("   Score:   " YEL "%d" RST, s->score);   break;
        case 3: printf("   Steps:   " YEL "%d" RST, s->steps);   break;
        case 5: printf("   Best:    " YEL "%d" RST, best_all);   break;
        case 7: printf("   Fitness: " YEL "%.0f" RST, best_fit); break;
        case 9: printf("   Length:  " YEL "%d" RST, s->len);     break;
        }
        printf("\n");
    }

    printf("  " CYN "+");
    for (c = 0; c < BW; c++) printf("--");
    printf("+" RST "\n\n");
    printf(DIM "  Net: %d->%d->%d  |  Pop: %d  |  Elite: %d  |  Ctrl+C to exit\n" RST,
           N_IN, N_HID, N_OUT, POP, ELITE);
    fflush(stdout);
}

/* ---- neuroevolution --------------------------------------------------- */
static int cmp_desc(const void *a, const void *b)
{
    const Snake *sa = (const Snake *)a;
    const Snake *sb = (const Snake *)b;
    if (sb->fitness > sa->fitness) return  1;
    if (sb->fitness < sa->fitness) return -1;
    return 0;
}

static void mutate(network_t *net, float rate, float strength)
{
    int l, nu, nl;
    for (l = 1; l < net->no_of_layers; l++)
        for (nu = 0; nu < net->layer[l].no_of_neurons; nu++)
            for (nl = 0; nl <= net->layer[l-1].no_of_neurons; nl++)
                if ((float)rand()/(float)RAND_MAX < rate) {
                    float u1 = (float)rand()/(float)RAND_MAX + 1e-7f;
                    float u2 = (float)rand()/(float)RAND_MAX;
                    float z  = (float)sqrt(-2.0*log((double)u1))
                             * (float)cos(6.28318530718*(double)u2);
                    net->layer[l].neuron[nu].weight[nl] += strength * z;
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

static void evolve_brains(void)
{
    int i, p1, p2;
    for (i = ELITE; i < POP; i++) {
        p1 = rand() % ELITE;
        p2 = (rand() % (ELITE - 1) + p1 + 1) % ELITE;
        crossover(pop[i].brain, pop[p1].brain, pop[p2].brain);
        mutate(pop[i].brain, 0.12f, 0.25f);
    }
    for (i = POP - 3; i < POP; i++)
        net_randomize(pop[i].brain, 1.0f);
}

/* ---- main ------------------------------------------------------------- */
int main(void)
{
    int i, alive, gen, best_all;
    float best_fit;
    int sizes[3];

    sizes[0] = N_IN; sizes[1] = N_HID; sizes[2] = N_OUT;

    srand((unsigned int)time(NULL));
    enable_vt100();
    HIDE();

    for (i = 0; i < POP; i++) {
        pop[i].brain = net_allocate_l(3, sizes);
        net_set_activation(pop[i].brain, NERVENET_ACTIVATION_TANH);
        net_initialize_xavier(pop[i].brain);
        reset_snake(&pop[i]);
    }

    best_all = 0;
    best_fit = 0.0f;

    for (gen = 1; ; gen++) {
        /* evaluate entire population */
        for (i = 0; i < POP; i++) reset_snake(&pop[i]);
        do {
            alive = 0;
            for (i = 0; i < POP; i++) {
                if (pop[i].alive) { step_snake(&pop[i]); alive += pop[i].alive; }
            }
        } while (alive > 0);

        for (i = 0; i < POP; i++) compute_fitness(&pop[i]);
        qsort(pop, POP, sizeof(Snake), cmp_desc);

        best_fit = pop[0].fitness;
        if (pop[0].score > best_all) best_all = pop[0].score;

        /* showcase best brain */
        reset_snake(&pop[0]);
        while (pop[0].alive) {
            step_snake(&pop[0]);
            draw(&pop[0], gen, best_all, best_fit);
            SLEEP_MS(FRAME_MS);
        }

        evolve_brains();
    }

    /* never reached during normal operation */
    for (i = 0; i < POP; i++) net_free(pop[i].brain);
    SHOW();
    return 0;
}
