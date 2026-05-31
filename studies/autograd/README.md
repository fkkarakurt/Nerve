# Nerve — autograd spike

A minimal, single-header, **reverse-mode automatic differentiation** engine over
2D tensors — ~200 lines, no dependencies. This is the spine that lets new layer
types be *expressed* instead of hand-derived, and it powers Nerve's on-device
learning (see `../infer/learn.c`).

Every operation records itself on a tape at forward time; `ng_backward()` walks
the tape in reverse and fills in each tensor's gradient. You compose forward ops
(`t_matmul`, `t_add_bias`, `t_relu`, `t_softmax_cross_entropy`); the chain rule
falls out. No gradient is ever written by hand.

## Files

| File | What it is |
|------|------------|
| `nerve_grad.h` | The autodiff engine (tensors, tape, ops, SGD). |
| `demo_mlp.c`   | Trains a 2-16-3 MLP to 12/12 with zero hand-written gradients. |
| `gradcheck.c`  | Verifies correctness: analytic gradients vs central finite differences agree to the float floor (~1e-4). |

## Run

```sh
gcc -O2 -std=c89 -Wall demo_mlp.c  -o demo_mlp  -lm && ./demo_mlp
gcc -O2 -std=c89 -Wall gradcheck.c -o gradcheck -lm && ./gradcheck
```

## Status & next

This is a **spike** (prototype), not yet folded into the shipped core. Current
ops cover MLP training. Next rungs toward generative on-device personalization
(LoRA): backward passes for attention, RMSNorm and RoPE so adapter matrices can
be trained through a frozen Transformer.
