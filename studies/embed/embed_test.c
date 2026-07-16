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

/* Quick quality probe for the MiniLM encoder: classify tricky inputs by nearest
 * category centroid (cosine). This is the "is it actually smart?" test. */
#define NERVE_EMBED_IMPLEMENTATION
#include "nerve_embed.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char *CAL[] = {"schedule a meeting tomorrow","move my dentist appointment",
    "remind me about the deadline","book a flight for monday","what time is my appointment"};
static const char *FOO[] = {"add tomatoes to the shopping list","how long to roast the chicken",
    "try a new pasta recipe","i am hungry for pizza","order some sushi tonight"};
static const char *FIT[] = {"i went for a long run","plan a gym session","do some push ups",
    "my legs are sore after the workout","sign up for a yoga class"};
static const char *NAMES[] = {"calendar","food","fitness"};

static const char *TESTS[] = {"cola","i want to eat a hamburger","what time is my appointment",
    "go jogging in the park","order a pizza","the sky is blue today","set a reminder for the call",
    "bench press personal best","i need new running shoes","buy milk and eggs"};

int main(int argc, char **argv)
{
    nerve_embed_t m;
    int H, i, j, c, rc;
    const char *mp = (argc > 1) ? argv[1] : "minilm.nre";
    const char **CATS[3]; int CN[3];
    float *cent;
    CATS[0]=CAL; CN[0]=5; CATS[1]=FOO; CN[1]=5; CATS[2]=FIT; CN[2]=5;

    if ((rc = nerve_embed_load(&m, mp, "vocab.txt"))) { printf("load %d\n", rc); return 1; }
    H = nerve_embed_dim(&m);
    printf("MiniLM encoder loaded: dim=%d, %d layers\n\n", H, m.cfg.n_layers);

    /* category centroids = normalised mean of anchor embeddings */
    cent = (float *)calloc((size_t)3 * H, sizeof(float));
    {
        float *e = (float *)malloc((size_t)H * sizeof(float));
        for (c = 0; c < 3; c++) {
            float nrm = 0;
            for (i = 0; i < CN[c]; i++) { nerve_embed_text(&m, CATS[c][i], e);
                for (j = 0; j < H; j++) cent[c*H+j] += e[j]; }
            for (j = 0; j < H; j++) cent[c*H+j] /= CN[c];
            for (j = 0; j < H; j++) nrm += cent[c*H+j]*cent[c*H+j];
            nrm = 1.0f/((float)sqrt((double)nrm)+1e-8f);
            for (j = 0; j < H; j++) cent[c*H+j] *= nrm;
        }
        free(e);
    }

    {
        float *e = (float *)malloc((size_t)H * sizeof(float));
        int N = (int)(sizeof(TESTS)/sizeof(TESTS[0]));
        for (i = 0; i < N; i++) {
            float sc[3], best=-2; int bi=0;
            nerve_embed_text(&m, TESTS[i], e);
            for (c = 0; c < 3; c++) { float s=0; for (j=0;j<H;j++) s+=e[j]*cent[c*H+j]; sc[c]=s; if(s>best){best=s;bi=c;} }
            printf("  %-9s [cal %.2f food %.2f fit %.2f]  \"%s\"\n",
                   NAMES[bi], sc[0], sc[1], sc[2], TESTS[i]);
        }
        free(e);
    }
    free(cent); nerve_embed_free(&m);
    return 0;
}
