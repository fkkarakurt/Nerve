#!/usr/bin/env python3
"""
convert_minilm.py — import a Hugging Face BERT/MiniLM sentence-embedding model
into Nerve's encoder format (.nre). Build-time tool, pure numpy + stdlib.

Writes a 64-byte header then float32 weights in a fixed canonical order that
nerve_embed.h reads back. The vocab.txt is used as-is (one WordPiece per line).

Usage: python convert_minilm.py minilm.safetensors minilm_config.json minilm.nre
"""
import json, struct, sys
import numpy as np

src  = sys.argv[1] if len(sys.argv) > 1 else "minilm.safetensors"
cfgp = sys.argv[2] if len(sys.argv) > 2 else "minilm_config.json"
outp = sys.argv[3] if len(sys.argv) > 3 else "minilm.nre"

cfg = json.load(open(cfgp))
H   = cfg["hidden_size"]
L   = cfg["num_hidden_layers"]
NH  = cfg["num_attention_heads"]
I   = cfg["intermediate_size"]
V   = cfg["vocab_size"]
MP  = cfg["max_position_embeddings"]

f = open(src, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
hdr  = json.loads(f.read(hlen))
DATA = 8 + hlen
def get(name):
    m = hdr[name]; s, e = m["data_offsets"]
    assert m["dtype"] == "F32", name + " " + m["dtype"]
    f.seek(DATA + s)
    return np.frombuffer(f.read(e - s), "<f4").reshape(m["shape"])

q8 = (len(sys.argv) > 4 and sys.argv[4] == "int8")

out = open(outp, "wb")
out.write(b"NEMB")
for v in (1, H, L, NH, I, V, MP, 1 if q8 else 0):    # version + 6 cfg + flags
    out.write(struct.pack("<i", v))
out.write(b"\x00" * (64 - 4 - 8*4))                  # pad header to 64 bytes

def wf(a): out.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
def wq(W):                                           # per-row symmetric int8
    amax = np.abs(W).max(axis=1)
    sc = np.where(amax > 0, amax / 127.0, 1.0).astype("<f4")
    out.write(sc.tobytes())
    out.write(np.clip(np.rint(W / sc[:, None]), -127, 127).astype(np.int8).tobytes())
W = wq if q8 else wf                                 # big matrices: quantised if int8

# embeddings
W(get("embeddings.word_embeddings.weight"))
wf(get("embeddings.position_embeddings.weight"))
wf(get("embeddings.token_type_embeddings.weight"))
wf(get("embeddings.LayerNorm.weight")); wf(get("embeddings.LayerNorm.bias"))
# layers
P = "encoder.layer.{}."
for l in range(L):
    p = P.format(l)
    W(get(p+"attention.self.query.weight")); wf(get(p+"attention.self.query.bias"))
    W(get(p+"attention.self.key.weight"));   wf(get(p+"attention.self.key.bias"))
    W(get(p+"attention.self.value.weight")); wf(get(p+"attention.self.value.bias"))
    W(get(p+"attention.output.dense.weight")); wf(get(p+"attention.output.dense.bias"))
    wf(get(p+"attention.output.LayerNorm.weight")); wf(get(p+"attention.output.LayerNorm.bias"))
    W(get(p+"intermediate.dense.weight")); wf(get(p+"intermediate.dense.bias"))
    W(get(p+"output.dense.weight")); wf(get(p+"output.dense.bias"))
    wf(get(p+"output.LayerNorm.weight")); wf(get(p+"output.LayerNorm.bias"))
out.close()
print(f"wrote {outp}  (H={H} L={L} heads={NH} I={I} vocab={V} {'int8' if q8 else 'fp32'})")
