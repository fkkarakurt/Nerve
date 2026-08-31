/* discover_web.c - the WebAssembly face of nerve_discover.h.
 *
 * Copyright 2022-2026 Fatih Kucukkarakurt <fatihkucukkarakurt@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * The whole engine, compiled to a WebAssembly module of a few tens of
 * kilobytes. There is no model to download, no weights, no data file: an
 * equation discoverer is pure algorithm, so the page is usable the moment it
 * loads and every byte of the computation happens on the visitor's machine.
 *
 * This translation unit is meant to run inside a Web Worker. The search is a
 * blocking loop, and a blocking loop on the main thread is a frozen tab; from
 * a worker it can report a new best equation after every generation while the
 * page stays live. Progress is pushed out through postMessage.
 *
 * Build: see build.sh / build.ps1 in this directory.
 */
#define NERVE_DISCOVER_IMPLEMENTATION
#include "../../nerve_discover.h"

#include <emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NDW_MAXVARS 12
#define NDW_MAXROWS 20000

static double  *g_X      = NULL;
static double  *g_y      = NULL;
static int      g_rows   = 0;
static int      g_nvars  = 0;
static char     g_names[NDW_MAXVARS][48];
static char    *g_namep[NDW_MAXVARS];
static char     g_error[256];

/* ---- reporting to JavaScript ------------------------------------------- */

EM_JS(void, ndw_post_progress, (int gen, int total, double nrmse,
                                const char *expr, int nodes), {
    postMessage({
        type: "progress",
        gen: gen, total: total,
        nrmse: nrmse, nodes: nodes,
        expr: expr ? UTF8ToString(expr) : ""
    });
});

EM_JS(void, ndw_post_front, (int nodes, double r2, const char *expr), {
    postMessage({
        type: "front",
        nodes: nodes, r2: r2, expr: UTF8ToString(expr)
    });
});

EM_JS(void, ndw_post_done, (int knee_nodes, double knee_r2,
                            const char *knee_expr, double seconds), {
    postMessage({
        type: "done",
        nodes: knee_nodes, r2: knee_r2,
        expr: UTF8ToString(knee_expr), seconds: seconds
    });
});

EM_JS(void, ndw_post_error, (const char *msg), {
    postMessage({ type: "error", message: UTF8ToString(msg) });
});

/* ---- CSV --------------------------------------------------------------- */

/* Accepts comma, semicolon, tab or whitespace separated numbers. A first row
 * that does not parse as numbers is taken as the header, and its fields name
 * the columns. The LAST column is the target; everything before it is an
 * input. Blank lines and lines starting with # are skipped. */

static int is_sep(char c){ return c == ',' || c == ';' || c == '\t' || c == ' '; }

static int split_line(char *line, char **field, int maxf){
    int n = 0;
    char *p = line;
    while (*p && n < maxf){
        char *start;
        while (*p && is_sep(*p)) p++;
        if (!*p) break;
        start = p;
        while (*p && !is_sep(*p)) p++;
        if (*p) *p++ = '\0';
        field[n++] = start;
    }
    return n;
}

static int all_numeric(char **field, int n){
    int i;
    for (i = 0; i < n; i++){
        char *end;
        strtod(field[i], &end);
        while (*end == '\r' || *end == '\n') end++;
        if (end == field[i] || *end != '\0') return 0;
    }
    return 1;
}

static void free_data(void){
    free(g_X); free(g_y);
    g_X = NULL; g_y = NULL; g_rows = 0; g_nvars = 0;
}

/* Returns the number of rows loaded, or 0 with g_error set. */
EMSCRIPTEN_KEEPALIVE
int ndw_load_csv(const char *text){
    char  *buf, *cur;
    char  *field[NDW_MAXVARS + 1];
    int    ncol = 0, row = 0, i, header_taken = 0;
    size_t len;

    free_data();
    g_error[0] = '\0';
    if (!text || !*text){ strcpy(g_error, "no data"); return 0; }

    len = strlen(text);
    buf = (char *)malloc(len + 1);
    if (!buf){ strcpy(g_error, "out of memory"); return 0; }
    memcpy(buf, text, len + 1);

    g_X = (double *)malloc(sizeof(double) * NDW_MAXROWS * NDW_MAXVARS);
    g_y = (double *)malloc(sizeof(double) * NDW_MAXROWS);
    if (!g_X || !g_y){ free(buf); free_data(); strcpy(g_error, "out of memory"); return 0; }

    /* Walk the lines by hand rather than with strtok_r, which is POSIX and
     * not guaranteed by C99 - the point of this project is to need nothing. */
    for (cur = buf; *cur; ){
        char *line = cur;
        int   n;
        while (*cur && *cur != '\n' && *cur != '\r') cur++;
        while (*cur == '\n' || *cur == '\r') *cur++ = '\0';
        if (!*line || *line == '#') continue;
        n = split_line(line, field, NDW_MAXVARS + 1);
        if (n < 2) continue;

        if (!header_taken){
            header_taken = 1;
            ncol = n;
            if (ncol > NDW_MAXVARS + 1) ncol = NDW_MAXVARS + 1;
            if (!all_numeric(field, n)){
                for (i = 0; i < ncol - 1; i++){
                    strncpy(g_names[i], field[i], sizeof g_names[0] - 1);
                    g_names[i][sizeof g_names[0] - 1] = '\0';
                }
                continue;                     /* it was a header, not a row */
            }
            for (i = 0; i < ncol - 1; i++)
                sprintf(g_names[i], "x%d", i);
        }
        if (n != ncol) continue;              /* ragged row: skip it        */
        if (row >= NDW_MAXROWS) break;

        for (i = 0; i < ncol - 1; i++)
            g_X[(size_t)row * (ncol - 1) + i] = strtod(field[i], NULL);
        g_y[row] = strtod(field[ncol - 1], NULL);
        {   /* drop any row that is not entirely finite */
            int ok = (g_y[row] == g_y[row]);
            for (i = 0; ok && i < ncol - 1; i++){
                double v = g_X[(size_t)row * (ncol - 1) + i];
                if (v != v) ok = 0;
            }
            if (ok) row++;
        }
    }
    free(buf);

    if (ncol < 2 || row < 4){
        free_data();
        strcpy(g_error, "need at least 4 rows and 2 columns "
                        "(inputs first, target last)");
        return 0;
    }
    g_nvars = ncol - 1;
    g_rows  = row;
    for (i = 0; i < g_nvars; i++) g_namep[i] = g_names[i];
    return g_rows;
}

EMSCRIPTEN_KEEPALIVE const char *ndw_error(void){ return g_error; }
EMSCRIPTEN_KEEPALIVE int ndw_rows(void){ return g_rows; }
EMSCRIPTEN_KEEPALIVE int ndw_vars(void){ return g_nvars; }
EMSCRIPTEN_KEEPALIVE const char *ndw_var_name(int i){
    return (i >= 0 && i < g_nvars) ? g_names[i] : "";
}

/* ---- the search -------------------------------------------------------- */

static char g_text[4096];

static int on_generation(int gen, int total, double nrmse,
                         const nd_expr *best, void *user){
    (void)user;
    /* Every generation would flood the message queue on an easy problem, and
     * every tenth is still smoother than the eye can follow. */
    if ((gen % 5) == 0 || gen == total){
        int nodes = 0;
        g_text[0] = '\0';
        if (best){
            nd_format(best, g_namep, g_text, (int)sizeof g_text);
            nodes = nd_complexity(best);
        }
        ndw_post_progress(gen, total, nrmse, g_text, nodes);
    }
    return 0;
}

EMSCRIPTEN_KEEPALIVE
void ndw_run(unsigned int ops, int generations, int max_nodes,
             int population, double seed){
    nd_options o;
    nd_model  *m;
    double     t0;
    int        i;

    if (!g_X || g_rows < 4){ ndw_post_error("load a dataset first"); return; }

    o = nd_defaults();
    o.ops         = ops ? ops : ND_OPS_DEFAULT;
    o.generations = generations > 0 ? generations : 200;
    o.max_nodes   = max_nodes  > 0 ? max_nodes  : 24;
    o.population  = population > 0 ? population : 300;
    o.islands     = 4;
    o.seed        = (unsigned long long)seed;
    o.progress    = on_generation;

    t0 = emscripten_get_now();
    m  = nd_fit(g_X, g_y, g_rows, g_nvars, o);
    if (!m){ ndw_post_error("the search could not be started"); return; }

    for (i = 0; i < nd_count(m); i++){
        const nd_expr *e = nd_at(m, i);
        nd_format(e, g_namep, g_text, (int)sizeof g_text);
        ndw_post_front(nd_complexity(e),
                       nd_r2(e, g_X, g_y, g_rows, g_nvars), g_text);
    }
    {
        const nd_expr *knee = nd_knee(m, 0.02);
        nd_format(knee, g_namep, g_text, (int)sizeof g_text);
        ndw_post_done(nd_complexity(knee),
                      nd_r2(knee, g_X, g_y, g_rows, g_nvars),
                      g_text, (emscripten_get_now() - t0) / 1000.0);
    }
    nd_free(m);
}

/* Evaluate the last reported equation is not needed by the page; the module
 * exists to produce formulas, and a formula is text the page can keep. */
int main(void){ return 0; }
