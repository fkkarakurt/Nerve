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

out = open(outp, "wb")
out.write(b"NEMB")
for v in (1, H, L, NH, I, V, MP):
    out.write(struct.pack("<i", v))
out.write(b"\x00" * (64 - 4 - 7*4))      # pad header to 64 bytes

def w(a): out.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

# embeddings
w(get("embeddings.word_embeddings.weight"))
w(get("embeddings.position_embeddings.weight"))
w(get("embeddings.token_type_embeddings.weight"))
w(get("embeddings.LayerNorm.weight")); w(get("embeddings.LayerNorm.bias"))
# layers
P = "encoder.layer.{}."
for l in range(L):
    p = P.format(l)
    w(get(p+"attention.self.query.weight")); w(get(p+"attention.self.query.bias"))
    w(get(p+"attention.self.key.weight"));   w(get(p+"attention.self.key.bias"))
    w(get(p+"attention.self.value.weight")); w(get(p+"attention.self.value.bias"))
    w(get(p+"attention.output.dense.weight")); w(get(p+"attention.output.dense.bias"))
    w(get(p+"attention.output.LayerNorm.weight")); w(get(p+"attention.output.LayerNorm.bias"))
    w(get(p+"intermediate.dense.weight")); w(get(p+"intermediate.dense.bias"))
    w(get(p+"output.dense.weight")); w(get(p+"output.dense.bias"))
    w(get(p+"output.LayerNorm.weight")); w(get(p+"output.LayerNorm.bias"))
out.close()
print(f"wrote {outp}  (H={H} L={L} heads={NH} I={I} vocab={V})")
