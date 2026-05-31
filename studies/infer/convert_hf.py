#!/usr/bin/env python3
"""
convert_hf.py - import a Hugging Face Llama-architecture model into Nerve's
native int8 .nrv format. BUILD-TIME tool only: Nerve's runtime (nerve_infer.h)
stays pure zero-dependency C. Uses just numpy + the stdlib (no torch, no
safetensors lib) - the .safetensors container is parsed by hand.

Handles: bf16/f16/f32 weights, the HF->interleaved RoPE un-permutation of the
q/k projections, GQA, and per-row symmetric int8 quantization written in the
exact layout nerve_infer.h's map_weights_q expects (all scales of a tensor
first, then all its int8 rows).

Usage:  python convert_hf.py tinyllama.safetensors config.json tinyllama_q8.nrv
"""
import json, struct, sys
import numpy as np

src   = sys.argv[1] if len(sys.argv) > 1 else "tinyllama.safetensors"
cfgp  = sys.argv[2] if len(sys.argv) > 2 else "tinyllama_config.json"
outp  = sys.argv[3] if len(sys.argv) > 3 else "tinyllama_q8.nrv"

cfg = json.load(open(cfgp))
dim     = cfg["hidden_size"]
hidden  = cfg["intermediate_size"]
L       = cfg["num_hidden_layers"]
n_heads = cfg["num_attention_heads"]
n_kv    = cfg["num_key_value_heads"]
vocab   = cfg["vocab_size"]
seq     = cfg["max_position_embeddings"]
rope    = float(cfg.get("rope_theta", 10000.0))
shared  = bool(cfg.get("tie_word_embeddings", False))
head_dim = dim // n_heads

print(f"config: dim={dim} hidden={hidden} L={L} heads={n_heads} kv={n_kv} "
      f"vocab={vocab} seq={seq} rope={rope} tied={shared}")

# ---- safetensors reader (header = u64 length, then JSON, then raw tensors) ----
f = open(src, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
header = json.loads(f.read(hlen))
DATA = 8 + hlen

def get(name):
    m = header[name]; s, e = m["data_offsets"]; dt = m["dtype"]
    f.seek(DATA + s); raw = f.read(e - s)
    if   dt == "BF16": a = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":  a = np.frombuffer(raw, "<f2").astype(np.float32)
    elif dt == "F32":  a = np.frombuffer(raw, "<f4").astype(np.float32)
    else: raise ValueError("dtype " + dt)
    return a.reshape(m["shape"])

def unpermute(w, nh):
    """HF (NeoX split-half) -> interleaved-pairs RoPE layout, on rows of [out,in]."""
    d1, d2 = w.shape
    return w.reshape(nh, 2, d1 // nh // 2, d2).swapaxes(1, 2).reshape(d1, d2)

out = open(outp, "wb")

# ---- 64-byte self-describing header ----
flags = (1 if shared else 0) | 2          # bit0 shared classifier, bit1 quantized
out.write(b"NRV1")
out.write(struct.pack("<i", 1))           # format version
for v in (dim, hidden, L, n_heads, n_kv, vocab, seq, flags):
    out.write(struct.pack("<i", v))
out.write(struct.pack("<f", rope))
out.write(struct.pack("<i", 0))           # reserved
out.write(b"\x00" * (64 - 48))

def quant_write(load, n):
    """Per-row symmetric int8: write every layer's scales first, then every
    layer's int8 rows - matching nerve_infer.h's map_weights_q."""
    scales = []
    for l in range(n):
        W = load(l)
        amax = np.abs(W).max(axis=1)
        sc = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        scales.append(sc)
        out.write(sc.tobytes())
    for l in range(n):
        W = load(l)
        q = np.rint(W / scales[l][:, None]).clip(-127, 127).astype(np.int8)
        out.write(q.tobytes())

def float_write(load, n):
    for l in range(n):
        out.write(load(l).astype(np.float32).tobytes())

P = "model.layers.{}."
quant_write(lambda _: get("model.embed_tokens.weight"), 1)                       # token_embedding
float_write(lambda l: get(P.format(l) + "input_layernorm.weight"), L)            # rms_att
quant_write(lambda l: unpermute(get(P.format(l) + "self_attn.q_proj.weight"), n_heads), L)  # wq
quant_write(lambda l: unpermute(get(P.format(l) + "self_attn.k_proj.weight"), n_kv), L)     # wk
quant_write(lambda l: get(P.format(l) + "self_attn.v_proj.weight"), L)           # wv
quant_write(lambda l: get(P.format(l) + "self_attn.o_proj.weight"), L)           # wo
float_write(lambda l: get(P.format(l) + "post_attention_layernorm.weight"), L)   # rms_ffn
quant_write(lambda l: get(P.format(l) + "mlp.gate_proj.weight"), L)              # w1
quant_write(lambda l: get(P.format(l) + "mlp.down_proj.weight"), L)              # w2
quant_write(lambda l: get(P.format(l) + "mlp.up_proj.weight"), L)                # w3
float_write(lambda _: get("model.norm.weight"), 1)                               # rms_final
if not shared:
    quant_write(lambda _: get("lm_head.weight"), 1)                              # wcls

out.close()
print("wrote", outp)
