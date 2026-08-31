/*
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
 */

/* nerve_discover.c - the npm package's equation-discovery entry point.
 *
 * One call in, one JSON document out: the whole Pareto front and the knee.
 * The data arrives as plain arrays already in the WebAssembly heap, because
 * parsing a CSV is something JavaScript does perfectly well and C should not
 * have to. There is no model to load and no state to keep between calls.
 *
 * Built by build.ps1 into npm/discover.mjs + npm/discover.wasm.
 */
#define NERVE_DISCOVER_IMPLEMENTATION
#include "../nerve_discover.h"

#include <emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NDJ_MAXVARS 32

/* The returned document is owned by this module and is valid until the next
 * call, which is the simplest contract that avoids handing JavaScript a
 * pointer it would have to remember to free. */
static char  *g_json = NULL;
static size_t g_cap  = 0;
static size_t g_len  = 0;

static void j_reserve(size_t extra){
    if (g_len + extra + 1 <= g_cap) return;
    while (g_len + extra + 1 > g_cap) g_cap = g_cap ? g_cap * 2 : 4096;
    g_json = (char *)realloc(g_json, g_cap);
}
static void j_raw(const char *s){
    size_t n = strlen(s);
    j_reserve(n);
    if (!g_json) return;
    memcpy(g_json + g_len, s, n);
    g_len += n;
    g_json[g_len] = '\0';
}
/* JSON string escaping. Variable names come from a caller-supplied header row,
 * so they can contain anything at all. */
static void j_str(const char *s){
    j_raw("\"");
    for (; *s; s++){
        char buf[8];
        unsigned char c = (unsigned char)*s;
        switch (c){
            case '"':  j_raw("\\\"");  break;
            case '\\': j_raw("\\\\");  break;
            case '\n': j_raw("\\n");   break;
            case '\r': j_raw("\\r");   break;
            case '\t': j_raw("\\t");   break;
            default:
                if (c < 0x20){ sprintf(buf, "\\u%04x", c); j_raw(buf); }
                else { buf[0] = (char)c; buf[1] = '\0'; j_raw(buf); }
        }
    }
    j_raw("\"");
}
static void j_num(double v){
    char buf[40];
    /* JSON has no NaN or Infinity; a non-finite score becomes null. */
    if (!(v == v) || v > 1e300 || v < -1e300){ j_raw("null"); return; }
    sprintf(buf, "%.10g", v);
    j_raw(buf);
}
static void j_int(int v){
    char buf[24];
    sprintf(buf, "%d", v);
    j_raw(buf);
}

/* Split a newline-separated list of column names. Returns how many were found;
 * missing ones are left NULL so nd_format falls back to x0, x1, ... */
static int split_names(char *joined, char **out, int max){
    int n = 0;
    char *p = joined;
    if (!joined) return 0;
    while (*p && n < max){
        out[n++] = p;
        while (*p && *p != '\n') p++;
        if (*p) *p++ = '\0';
    }
    return n;
}

/* X is row-major (n * nvars doubles), y is n doubles, both already in the
 * heap. `names_joined` may be NULL. Returns a JSON document. */
EMSCRIPTEN_KEEPALIVE
const char *nd_json_fit(const double *X, const double *y, int n, int nvars,
                        char *names_joined, unsigned int ops, int generations,
                        int max_nodes, int population, double seed,
                        double knee_slack){
    char       *names[NDJ_MAXVARS];
    nd_options  o;
    nd_model   *m;
    char        text[4096];
    int         i, nn;

    g_len = 0;
    j_reserve(1024);
    if (g_json) g_json[0] = '\0';

    if (!X || !y || n < 4 || nvars < 1 || nvars > NDJ_MAXVARS){
        j_raw("{\"error\":\"need at least 4 rows and 1 to 32 inputs\"}");
        return g_json ? g_json : "{}";
    }

    for (i = 0; i < NDJ_MAXVARS; i++) names[i] = NULL;
    nn = split_names(names_joined, names, nvars);
    for (i = nn; i < nvars; i++) names[i] = NULL;

    o = nd_defaults();
    o.ops         = ops ? ops : ND_OPS_DEFAULT;
    o.generations = generations > 0 ? generations : 250;
    o.max_nodes   = max_nodes   > 0 ? max_nodes   : 24;
    o.population  = population  > 0 ? population  : 300;
    o.seed        = (unsigned long long)seed;

    m = nd_fit(X, y, n, nvars, o);
    if (!m){
        j_raw("{\"error\":\"the search could not be started\"}");
        return g_json ? g_json : "{}";
    }

    j_raw("{\"front\":[");
    for (i = 0; i < nd_count(m); i++){
        const nd_expr *e = nd_at(m, i);
        nd_format(e, names, text, (int)sizeof text);
        if (i) j_raw(",");
        j_raw("{\"nodes\":");   j_int(nd_complexity(e));
        j_raw(",\"r2\":");      j_num(nd_r2(e, X, y, n, nvars));
        j_raw(",\"rmse\":");    j_num(nd_rmse(e, X, y, n, nvars));
        j_raw(",\"equation\":");j_str(text);
        j_raw("}");
    }
    j_raw("],\"knee\":");
    {
        const nd_expr *k = nd_knee(m, knee_slack >= 0.0 ? knee_slack : 0.02);
        nd_format(k, names, text, (int)sizeof text);
        j_raw("{\"nodes\":");   j_int(nd_complexity(k));
        j_raw(",\"r2\":");      j_num(nd_r2(k, X, y, n, nvars));
        j_raw(",\"rmse\":");    j_num(nd_rmse(k, X, y, n, nvars));
        j_raw(",\"equation\":");j_str(text);
        j_raw("}");
    }
    j_raw("}");

    nd_free(m);
    return g_json ? g_json : "{}";
}

EMSCRIPTEN_KEEPALIVE const char *nd_json_version(void){ return ND_VERSION; }

int main(void){ return 0; }
