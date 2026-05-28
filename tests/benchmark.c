/*
 * Nerve v2.0.0 — Convergence & Accuracy Benchmark
 *
 * Compares SGD vs Adam, multiple activation functions, and weight
 * initialisation strategies on classic boolean and parity tasks.
 *
 * Build:  (handled by tests/Makefile)
 * Run:    ./benchmark
 */

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nerveNet.h"

/* ── Timing ─────────────────────────────────────────────────────────── */
static double elapsed_ms(clock_t start, clock_t end)
{
    return 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;
}

/* ── Problem Definition ─────────────────────────────────────────────── */
typedef struct
{
    const char *name;
    int         n_inputs;
    int         n_outputs;
    int         n_pairs;
    float      *inputs;    /* flat: n_pairs * n_inputs  */
    float      *targets;   /* flat: n_pairs * n_outputs */
    int         hidden;    /* hidden neurons             */
    float       tol;       /* MSE convergence threshold  */
    int         max_iters; /* iteration budget           */
} problem_t;

/* ── Configuration ─────────────────────────────────────────────────── */
typedef struct
{
    const char              *name;
    nervenet_activation_t    activation;
    nervenet_optimizer_t     optimizer;
    nervenet_init_t          init;
    float                    learning_rate;
    float                    momentum;
    float                    l2_lambda;
} config_t;

/* ── Result ─────────────────────────────────────────────────────────── */
typedef struct
{
    int   iters;       /* iterations until convergence; -1 = did not converge */
    float final_mse;
    float ms;          /* wall-clock time                */
} result_t;

/* ── Build & configure a network ───────────────────────────────────────*/
static network_t *make_net(const problem_t *p, const config_t *c)
{
    network_t *net = net_allocate(3, p->n_inputs, p->hidden, p->n_outputs);

    net_set_activation(net, c->activation);
    net_set_optimizer(net, c->optimizer);
    net_set_l2_lambda(net, c->l2_lambda);
    net_set_learning_rate(net, c->learning_rate);
    net_set_momentum(net, c->momentum);

    switch (c->init)
    {
    case NERVENET_INIT_XAVIER: net_initialize_xavier(net); break;
    case NERVENET_INIT_HE:     net_initialize_he(net);     break;
    default:                   net_randomize(net, 1.0f);   break;
    }
    return net;
}

/* ── Run one configuration on one problem ──────────────────────────── */
static result_t run_once(const problem_t *p, const config_t *c)
{
    network_t *net = make_net(p, c);
    result_t   r;
    int        i, it;
    float      mse, total_mse = 0.0f;
    clock_t    t0, t1;

    t0 = clock();

    for (it = 0; it < p->max_iters; it++)
    {
        /* one full epoch */
        total_mse = 0.0f;
        for (i = 0; i < p->n_pairs; i++)
        {
            net_compute(net, p->inputs  + i * p->n_inputs, NULL);
            mse = net_compute_output_error(net, p->targets + i * p->n_outputs);
            net_train(net);
            total_mse += mse;
        }
        total_mse /= (float)p->n_pairs;

        if (total_mse < p->tol)
        {
            t1       = clock();
            r.iters  = it + 1;
            r.final_mse = total_mse;
            r.ms     = (float)elapsed_ms(t0, t1);
            net_free(net);
            return r;
        }
    }

    t1        = clock();
    r.iters   = -1;
    r.final_mse = total_mse;
    r.ms      = (float)elapsed_ms(t0, t1);
    net_free(net);
    return r;
}

/* ── Multi-run helper (average over N_RUNS seeds) ───────────────────── */
#define N_RUNS 7

typedef struct
{
    int   successes;
    float avg_iters;
    float avg_mse;
    float avg_ms;
    int   best_iters;
} summary_t;

static summary_t run_multi(const problem_t *p, const config_t *c)
{
    summary_t s;
    result_t  r;
    int       k;
    long      iters_sum = 0;
    double    mse_sum   = 0.0, ms_sum = 0.0;

    memset(&s, 0, sizeof(s));
    s.best_iters = p->max_iters + 1;

    for (k = 0; k < N_RUNS; k++)
    {
        r = run_once(p, c);
        mse_sum += r.final_mse;
        ms_sum  += r.ms;
        if (r.iters > 0)
        {
            s.successes++;
            iters_sum += r.iters;
            if (r.iters < s.best_iters) s.best_iters = r.iters;
        }
    }

    s.avg_mse = (float)(mse_sum / N_RUNS);
    s.avg_ms  = (float)(ms_sum  / N_RUNS);
    s.avg_iters = (s.successes > 0)
                  ? (float)iters_sum / (float)s.successes
                  : (float)p->max_iters;
    return s;
}

/* ── Table printing ─────────────────────────────────────────────────── */
static void print_sep(int w)
{
    int i;
    for (i = 0; i < w; i++) putchar('-');
    putchar('\n');
}

static void print_header(void)
{
    printf("%-28s %8s %10s %8s %8s %8s\n",
           "Configuration", "Conv", "Avg Iters", "Best", "MSE", "ms");
}

static void print_row(const char *name, const summary_t *s, int n_runs,
                      int baseline_iters)
{
    float speedup;
    if (s->successes == 0)
        printf("%-28s %3d/%-4d %10s %8s %8.6f %8.1f\n",
               name, 0, n_runs, "—", "—", s->avg_mse, s->avg_ms);
    else
    {
        speedup = (baseline_iters > 0)
                  ? (float)baseline_iters / s->avg_iters
                  : 1.0f;
        printf("%-28s %3d/%-4d %10.0f %8d %8.6f %8.1f",
               name, s->successes, n_runs,
               s->avg_iters, s->best_iters, s->avg_mse, s->avg_ms);
        if (speedup > 1.05f)
            printf("  (%.1fx faster)", speedup);
        putchar('\n');
    }
}

/* ── Dataset helpers ────────────────────────────────────────────────── */
static problem_t make_xor(void)
{
    static float inp[] = {1,1, 1,0, 0,1, 0,0};
    static float tgt[] = {0,   1,   1,   0  };
    problem_t p;
    p.name      = "XOR";
    p.n_inputs  = 2; p.n_outputs = 1;
    p.n_pairs   = 4;
    p.inputs    = inp;  p.targets = tgt;
    p.hidden    = 4;
    p.tol       = 1e-4f;
    p.max_iters = 50000;
    return p;
}

static problem_t make_and(void)
{
    static float inp[] = {1,1, 1,0, 0,1, 0,0};
    static float tgt[] = {1,   0,   0,   0  };
    problem_t p;
    p.name      = "AND";
    p.n_inputs  = 2; p.n_outputs = 1;
    p.n_pairs   = 4;
    p.inputs    = inp;  p.targets = tgt;
    p.hidden    = 3;
    p.tol       = 1e-4f;
    p.max_iters = 10000;
    return p;
}

/* 3-bit parity: output = XOR of all 3 bits */
static problem_t make_parity3(void)
{
    static float inp[] = {
        0,0,0,  0,0,1,  0,1,0,  0,1,1,
        1,0,0,  1,0,1,  1,1,0,  1,1,1
    };
    static float tgt[] = {0, 1, 1, 0, 1, 0, 0, 1};
    problem_t p;
    p.name      = "Parity-3";
    p.n_inputs  = 3; p.n_outputs = 1;
    p.n_pairs   = 8;
    p.inputs    = inp;  p.targets = tgt;
    p.hidden    = 8;
    p.tol       = 1e-4f;
    p.max_iters = 200000;
    return p;
}

/* 4-class digit pattern (0–3) encoded as 2×2 patterns */
static problem_t make_digit4(void)
{
    /* Stylised 2x2 binary patterns → one-hot 4-class */
    static float inp[] = {
        1,0,0,0,   /* 0 */
        0,1,0,0,   /* 1 */
        0,0,1,0,   /* 2 */
        0,0,0,1    /* 3 */
    };
    static float tgt[] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    problem_t p;
    p.name      = "4-Class Identity";
    p.n_inputs  = 4; p.n_outputs = 4;
    p.n_pairs   = 4;
    p.inputs    = inp;  p.targets = tgt;
    p.hidden    = 6;
    p.tol       = 1e-4f;
    p.max_iters = 30000;
    return p;
}

/* ── Configurations to benchmark ────────────────────────────────────── */
static void run_problem(const problem_t *p, config_t *cfgs, int n_cfgs)
{
    summary_t results[32];
    int k, baseline = -1;
    const int W = 76;

    print_sep(W);
    printf("  Problem: %s   (arch: %d-%d-%d,  tol=%.0e,  %d runs)\n",
           p->name, p->n_inputs, p->hidden, p->n_outputs,
           (double)p->tol, N_RUNS);
    print_sep(W);
    print_header();
    print_sep(W);

    for (k = 0; k < n_cfgs; k++)
    {
        results[k] = run_multi(p, &cfgs[k]);
        if (k == 0 && results[0].successes > 0)
            baseline = (int)results[0].avg_iters;
        print_row(cfgs[k].name, &results[k], N_RUNS, baseline);
    }

    print_sep(W);
    printf("\n");
}

/* ── Main ────────────────────────────────────────────────────────────── */
int main(void)
{
    problem_t problems[4];
    config_t  cfgs[6];
    int       n_cfgs = 6;

    srand((unsigned int)time(NULL));

    /* ── Print banner ── */
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════╗\n");
    printf("  ║       Nerve %s  —  Neural Network Benchmark       ║\n",
           net_get_version());
    printf("  ║   SGD vs Adam · Sigmoid / Tanh / ReLU · Init study  ║\n");
    printf("  ╚══════════════════════════════════════════════════════╝\n\n");

    /* ── Define configurations ── */

    /* SGD + Uniform + Sigmoid  (original baseline) */
    cfgs[0].name          = "SGD  / Uniform / Sigmoid";
    cfgs[0].activation    = NERVENET_ACTIVATION_SIGMOID;
    cfgs[0].optimizer     = NERVENET_OPTIMIZER_SGD;
    cfgs[0].init          = NERVENET_INIT_UNIFORM;
    cfgs[0].learning_rate = 0.25f;
    cfgs[0].momentum      = 0.1f;
    cfgs[0].l2_lambda     = 0.0f;

    /* SGD + Xavier + Sigmoid */
    cfgs[1].name          = "SGD  / Xavier  / Sigmoid";
    cfgs[1].activation    = NERVENET_ACTIVATION_SIGMOID;
    cfgs[1].optimizer     = NERVENET_OPTIMIZER_SGD;
    cfgs[1].init          = NERVENET_INIT_XAVIER;
    cfgs[1].learning_rate = 0.25f;
    cfgs[1].momentum      = 0.1f;
    cfgs[1].l2_lambda     = 0.0f;

    /* SGD + Xavier + Tanh */
    cfgs[2].name          = "SGD  / Xavier  / Tanh";
    cfgs[2].activation    = NERVENET_ACTIVATION_TANH;
    cfgs[2].optimizer     = NERVENET_OPTIMIZER_SGD;
    cfgs[2].init          = NERVENET_INIT_XAVIER;
    cfgs[2].learning_rate = 0.10f;
    cfgs[2].momentum      = 0.1f;
    cfgs[2].l2_lambda     = 0.0f;

    /* Adam + Xavier + Sigmoid */
    cfgs[3].name          = "Adam / Xavier  / Sigmoid";
    cfgs[3].activation    = NERVENET_ACTIVATION_SIGMOID;
    cfgs[3].optimizer     = NERVENET_OPTIMIZER_ADAM;
    cfgs[3].init          = NERVENET_INIT_XAVIER;
    cfgs[3].learning_rate = 0.01f;
    cfgs[3].momentum      = 0.0f;
    cfgs[3].l2_lambda     = 0.0f;

    /* Adam + Xavier + Tanh */
    cfgs[4].name          = "Adam / Xavier  / Tanh";
    cfgs[4].activation    = NERVENET_ACTIVATION_TANH;
    cfgs[4].optimizer     = NERVENET_OPTIMIZER_ADAM;
    cfgs[4].init          = NERVENET_INIT_XAVIER;
    cfgs[4].learning_rate = 0.005f;
    cfgs[4].momentum      = 0.0f;
    cfgs[4].l2_lambda     = 0.0f;

    /* Adam + He + ReLU */
    cfgs[5].name          = "Adam / He      / ReLU";
    cfgs[5].activation    = NERVENET_ACTIVATION_RELU;
    cfgs[5].optimizer     = NERVENET_OPTIMIZER_ADAM;
    cfgs[5].init          = NERVENET_INIT_HE;
    cfgs[5].learning_rate = 0.005f;
    cfgs[5].momentum      = 0.0f;
    cfgs[5].l2_lambda     = 0.0f;

    /* ── Run each problem ── */
    problems[0] = make_xor();
    problems[1] = make_and();
    problems[2] = make_parity3();
    problems[3] = make_digit4();

    run_problem(&problems[0], cfgs, n_cfgs);
    run_problem(&problems[1], cfgs, n_cfgs);
    run_problem(&problems[2], cfgs, n_cfgs);
    run_problem(&problems[3], cfgs, n_cfgs);

    /* ── net_validate smoke test ── */
    {
        network_t *net;
        printf("  Validation checks:\n");

        net = net_allocate(3, 2, 4, 1);
        printf("    net_validate (fresh SGD net) ......... %s\n",
               net_validate(net) ? "PASS" : "FAIL");

        net_set_optimizer(net, NERVENET_OPTIMIZER_ADAM);
        printf("    net_validate (after Adam init) ....... %s\n",
               net_validate(net) ? "PASS" : "FAIL");

        {
            network_t *cp = net_copy(net);
            printf("    net_copy + validate .................. %s\n",
                   net_validate(cp) ? "PASS" : "FAIL");
            net_free(cp);
        }

        printf("    net_get_version() .................... %s\n",
               net_get_version());
        printf("    net_get_no_of_weights(2-4-1) ......... %d\n",
               net_get_no_of_weights(net));

        net_free(net);
    }

    printf("\n  Done.\n\n");
    return 0;
}
