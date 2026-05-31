/*
 * nerve_web.c — WebAssembly entry points for the browser demo.
 *
 * Reuses the unchanged engine (studies/infer/nerve_infer.h). The model and
 * tokenizer are preloaded into Emscripten's in-memory filesystem at build time,
 * so the existing fopen-based loaders work as-is — the same C that runs in the
 * terminal now runs in a browser tab, with no server and no GPU.
 */
#define NERVE_INFER_IMPLEMENTATION
#include "../studies/infer/nerve_infer.h"
#include <emscripten.h>

static nerve_transformer g_model;
static nerve_tokenizer   g_tok;
static int               g_ready = 0;

/* Push one decoded text fragment up to the page (defined in JS). */
EM_JS(void, nerve_js_emit, (const char *p), {
    if (Module.onPiece) Module.onPiece(UTF8ToString(p));
});

static void on_piece(const char *piece, void *user)
{
    (void)user;
    nerve_js_emit(piece);
}

/* Load the preloaded model + tokenizer. Returns 0 on success. */
EMSCRIPTEN_KEEPALIVE
int nerve_web_init(void)
{
    int rc;
    if ((rc = nerve_infer_load(&g_model, "model_q8.nrv")) != 0) return 10 + rc;
    if ((rc = nerve_tokenizer_load(&g_tok, "nerve.tok"))   != 0) return 20 + rc;
    g_ready = 1;
    return 0;
}

EMSCRIPTEN_KEEPALIVE
int nerve_web_ctx(void) { return g_ready ? g_model.config.seq_len : 0; }

EMSCRIPTEN_KEEPALIVE
int nerve_web_dim(void) { return g_ready ? g_model.config.dim : 0; }

/* Encode text -> the model's final hidden state (dim floats). Returns a pointer
 * into the engine's buffer (valid until the next call); JS copies it out. This
 * is the feature the in-browser "teach" demo trains a tiny head on. */
/* Mean-pooled sentence embedding (average of every token's final hidden state)
 * — a far better whole-sentence representation than the last token alone. */
static float g_embuf[8192];
EMSCRIPTEN_KEEPALIVE
float *nerve_web_embed(const char *text)
{
    int toks[1024], n, p, i, dim = g_model.config.dim;
    if (!g_ready) return 0;
    n = nerve_tokenizer_encode(&g_tok, text, 1, 0, toks);
    for (i = 0; i < dim; i++) g_embuf[i] = 0.0f;
    for (p = 0; p < n; p++) {
        float *h = (nerve_infer_forward(&g_model, toks[p], p), nerve_infer_hidden(&g_model));
        for (i = 0; i < dim; i++) g_embuf[i] += h[i];
    }
    for (i = 0; i < dim; i++) g_embuf[i] /= (float)(n > 0 ? n : 1);
    return g_embuf;
}

/* ---- streaming generation: one token per call, driven from JS so the page
 *      stays responsive and the text appears live ---- */
static int           g_ptoks[4096];
static int           g_np, g_pos, g_steps, g_token;
static nerve_sampler g_gs;
static char          g_piece[1024];

EMSCRIPTEN_KEEPALIVE
void nerve_web_gen_start(const char *prompt, int steps, float temp, float topp, int seed)
{
    if (!g_ready) return;
    g_np = nerve_tokenizer_encode(&g_tok, prompt, 1, 0, g_ptoks);
    if (g_np < 1) { g_pos = g_steps = 0; return; }
    if (steps > g_model.config.seq_len) steps = g_model.config.seq_len;
    g_steps = steps; g_pos = 0; g_token = g_ptoks[0];
    nerve_sampler_init(&g_gs, g_model.config.vocab_size, temp, topp,
                       (unsigned long long)(unsigned)seed);
}

EMSCRIPTEN_KEEPALIVE
int nerve_web_gen_done(void) { return (!g_ready) || g_pos >= g_steps; }

/* Advance one token; returns its decoded text fragment ("" when finished). */
EMSCRIPTEN_KEEPALIVE
const char *nerve_web_gen_step(void)
{
    float *logits; int next, prev; const char *piece;
    g_piece[0] = '\0';
    if (!g_ready || g_pos >= g_steps) { g_pos = g_steps; return g_piece; }
    logits = nerve_infer_forward(&g_model, g_token, g_pos);
    if (g_pos < g_np - 1) next = g_ptoks[g_pos + 1];
    else                  next = nerve_i__sample(&g_gs, logits);
    prev = g_token; g_pos++;
    if (next == 1) { g_pos = g_steps; nerve_sampler_free(&g_gs); return g_piece; }
    piece = nerve_i__decode(&g_tok, prev, next);
    strncpy(g_piece, piece, sizeof(g_piece) - 1);
    g_token = next;
    if (g_pos >= g_steps) nerve_sampler_free(&g_gs);
    return g_piece;
}

/* Generate from a prompt; each token fragment is streamed via Module.onPiece. */
EMSCRIPTEN_KEEPALIVE
void nerve_web_generate(const char *prompt, int steps, float temp, float topp, int seed)
{
    nerve_sampler s;
    if (!g_ready) return;
    if (steps > g_model.config.seq_len) steps = g_model.config.seq_len;
    nerve_sampler_init(&s, g_model.config.vocab_size, temp, topp,
                       (unsigned long long)(unsigned)seed);
    nerve_generate(&g_model, &g_tok, &s, prompt, steps, on_piece, 0);
    nerve_sampler_free(&s);
}
