/*
 * generate.c — proof: a real Transformer generating text, single-header, zero deps.
 *
 * Reads Nerve's own formats: model.nrv + nerve.tok (produce them with `convert`).
 *
 * Build:  gcc -O3 -march=native generate.c -o generate -lm
 * Run:    ./generate "Once upon a time"
 *         ./generate "Once upon a time" 256 0.9 0.9 42
 *           args: [prompt] [steps] [temperature] [top-p] [seed]
 */
#define NERVE_INFER_IMPLEMENTATION
#include "nerve_infer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    const char *prompt = (argc > 1) ? argv[1] : "Once upon a time";
    int    steps = (argc > 2) ? atoi(argv[2]) : 256;
    float  temp  = (argc > 3) ? (float)atof(argv[3]) : 0.9f;
    float  topp  = (argc > 4) ? (float)atof(argv[4]) : 0.9f;
    unsigned long long seed = (argc > 5) ? strtoull(argv[5], NULL, 10) : 42ULL;

    nerve_transformer model;
    nerve_tokenizer   tok;
    nerve_sampler     sampler;
    clock_t t0;
    int rc;

    {
        const char *mp = getenv("NERVE_MODEL");
        if (!mp) mp = "model.nrv";
        if ((rc = nerve_infer_load(&model, mp)) != 0) {
            fprintf(stderr, "failed to load %s (%d)\n", mp, rc); return 1;
        }
    }
    if (steps > model.config.seq_len) steps = model.config.seq_len;
    if ((rc = nerve_tokenizer_load(&tok, "nerve.tok")) != 0) {
        fprintf(stderr, "failed to load nerve.tok (%d)\n", rc); return 1;
    }
    nerve_sampler_init(&sampler, model.config.vocab_size, temp, topp, seed);

    printf("Nerve inference  |  dim=%d layers=%d heads=%d vocab=%d ctx=%d  |  single header, zero deps\n",
           model.config.dim, model.config.n_layers, model.config.n_heads,
           model.config.vocab_size, model.config.seq_len);
    printf("prompt: \"%s\"   (temp=%.2f top-p=%.2f seed=%llu)\n\n", prompt, temp, topp, seed);

    t0 = clock();
    nerve_generate(&model, &tok, &sampler, prompt, steps, NULL, NULL);
    {
        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("\n\n[%d steps in %.2fs  =>  %.1f tokens/sec]\n",
               steps, secs, secs > 0 ? steps / secs : 0.0);
    }

    nerve_sampler_free(&sampler);
    nerve_tokenizer_free(&tok);
    nerve_infer_free(&model);
    return 0;
}
