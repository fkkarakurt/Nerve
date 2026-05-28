"""
Nerve — Example 04: Python Binding via ctypes
Copyright (c) 2022 Fatih Küçükkarakurt <fatihkucukkarakurt@gmail.com>
SPDX-License-Identifier: MIT

Demonstrates calling the Nerve shared library from Python using ctypes.
No third-party Python packages required — only the standard library.

Build the shared library first:
  Linux/Mac:
    gcc -O2 -shared -fPIC -o libnerve.so \\
        -DNERVE_IMPLEMENTATION ../nerve.h -lm

  Windows (MinGW):
    gcc -O2 -shared -o nerve.dll \\
        -DNERVE_IMPLEMENTATION ../nerve.h -lm

Run:
  python 04_ctypes_demo.py
"""

import ctypes
import os
import sys
import platform

# ── Load shared library ──────────────────────────────────────────────────
def _find_lib():
    if platform.system() == "Windows":
        names = ["nerve.dll", "libnerve.dll"]
    elif platform.system() == "Darwin":
        names = ["libnerve.dylib", "libnerve.so"]
    else:
        names = ["libnerve.so"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in names:
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Nerve shared library not found.\n"
        "Build it with:\n"
        "  gcc -O2 -shared -fPIC -DNERVE_IMPLEMENTATION -o libnerve.so ../nerve.h -lm\n"
    )

lib = ctypes.CDLL(_find_lib())

# ── Type aliases ──────────────────────────────────────────────────────────
c_float_p = ctypes.POINTER(ctypes.c_float)

# opaque pointer — we never inspect network_t internals from Python
class NetworkPtr(ctypes.c_void_p):
    pass

# ── Bind functions ────────────────────────────────────────────────────────
lib.net_allocate.restype  = NetworkPtr
lib.net_allocate.argtypes = [ctypes.c_int, ctypes.c_int]  # variadic: works for 2-layer

lib.net_allocate_l.restype  = NetworkPtr
lib.net_allocate_l.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

lib.net_free.argtypes = [NetworkPtr]

lib.net_set_optimizer.argtypes     = [NetworkPtr, ctypes.c_int]
lib.net_set_activation.argtypes    = [NetworkPtr, ctypes.c_int]
lib.net_set_learning_rate.argtypes = [NetworkPtr, ctypes.c_float]
lib.net_initialize_xavier.argtypes = [NetworkPtr]
lib.net_initialize_he.argtypes     = [NetworkPtr]

lib.net_compute.argtypes              = [NetworkPtr, c_float_p, c_float_p]
lib.net_compute_output_error.restype  = ctypes.c_float
lib.net_compute_output_error.argtypes = [NetworkPtr, c_float_p]
lib.net_get_output_error.restype      = ctypes.c_float
lib.net_get_output_error.argtypes     = [NetworkPtr]

lib.net_train.argtypes = [NetworkPtr]

lib.net_classify.restype  = ctypes.c_int
lib.net_classify.argtypes = [NetworkPtr, c_float_p]

lib.net_compute_accuracy.restype  = ctypes.c_float
lib.net_compute_accuracy.argtypes = [NetworkPtr, c_float_p, c_float_p,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.net_validate.restype  = ctypes.c_int
lib.net_validate.argtypes = [NetworkPtr]

lib.net_get_version.restype = ctypes.c_char_p

lib.net_get_no_of_weights.restype  = ctypes.c_int
lib.net_get_no_of_weights.argtypes = [NetworkPtr]

# Optimiser and activation enum values
OPTIMIZER_SGD  = 0
OPTIMIZER_ADAM = 1
ACTIVATION_SIGMOID    = 0
ACTIVATION_TANH       = 1
ACTIVATION_RELU       = 2
ACTIVATION_LEAKY_RELU = 3


# ── High-level Python wrapper ─────────────────────────────────────────────
def floats(*values):
    """Convert a sequence of numbers to a ctypes float array."""
    arr = (ctypes.c_float * len(values))(*values)
    return arr


def floats_from_list(lst):
    arr = (ctypes.c_float * len(lst))(*lst)
    return arr


class NerveNet:
    """Thin Python wrapper around a Nerve network."""

    def __init__(self, *layer_sizes):
        layers = (ctypes.c_int * len(layer_sizes))(*layer_sizes)
        self._net = lib.net_allocate_l(len(layer_sizes), layers)
        if not self._net:
            raise MemoryError("net_allocate_l returned NULL")
        self._n_in  = layer_sizes[0]
        self._n_out = layer_sizes[-1]

    def __del__(self):
        if self._net:
            lib.net_free(self._net)

    def use_adam(self, lr=0.01):
        lib.net_set_optimizer(self._net, OPTIMIZER_ADAM)
        lib.net_set_learning_rate(self._net, ctypes.c_float(lr))

    def use_sgd(self, lr=0.25):
        lib.net_set_optimizer(self._net, OPTIMIZER_SGD)
        lib.net_set_learning_rate(self._net, ctypes.c_float(lr))

    def activation(self, act):
        lib.net_set_activation(self._net, act)

    def xavier_init(self):
        lib.net_initialize_xavier(self._net)

    def predict(self, x):
        inp = floats_from_list(x)
        out = (ctypes.c_float * self._n_out)()
        lib.net_compute(self._net, inp, out)
        return list(out)

    def train_step(self, x, y):
        inp = floats_from_list(x)
        tgt = floats_from_list(y)
        lib.net_compute(self._net, inp, None)
        mse = lib.net_compute_output_error(self._net, tgt)
        lib.net_train(self._net)
        return float(mse)

    def classify(self, x):
        inp = floats_from_list(x)
        return lib.net_classify(self._net, inp)

    @property
    def n_weights(self):
        return lib.net_get_no_of_weights(self._net)

    def is_valid(self):
        return bool(lib.net_validate(self._net))


# ── Demo: XOR with Python API ─────────────────────────────────────────────
def demo_xor():
    import random, time
    random.seed(42)

    version = lib.net_get_version().decode()
    print(f"Nerve {version} — Python ctypes Demo")
    print("=" * 45)
    print(f"\n[XOR]  Architecture: 2-4-1 | Adam | Xavier\n")

    net = NerveNet(2, 4, 1)
    net.use_adam(lr=0.01)
    net.xavier_init()
    print(f"  Weights: {net.n_weights}   Valid: {net.is_valid()}\n")

    data = [
        ([1, 1], [0]),
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([0, 0], [0]),
    ]

    t0 = time.time()
    for epoch in range(1, 10001):
        mse_sum = 0.0
        for x, y in data:
            mse_sum += net.train_step(x, y)
        mse = mse_sum / len(data)
        if epoch % 2000 == 0:
            print(f"  Epoch {epoch:5d}  |  MSE: {mse:.6f}")
        if mse < 1e-4:
            print(f"  Converged at epoch {epoch}  |  MSE: {mse:.6f}")
            break
    elapsed = time.time() - t0

    print(f"\n  Results (elapsed: {elapsed*1000:.0f} ms):")
    print(f"  {'Input':<12}  {'Target':<8}  {'Output':<10}  OK?")
    for x, y in data:
        out = net.predict(x)[0]
        ok  = "✓" if (out > 0.5) == bool(y[0]) else "✗"
        print(f"  {str(x):<12}  {y[0]:<8.0f}  {out:<10.4f}  {ok}")


# ── Demo: sin(x) with mini-batch ─────────────────────────────────────────
def demo_sine():
    import math, random
    random.seed(0)
    print("\n[sin(x)] Architecture: 1-16-16-1 | Adam + Tanh\n")

    net = NerveNet(1, 16, 16, 1)
    net.use_adam(lr=0.005)
    net.activation(ACTIVATION_TANH)
    net.xavier_init()

    # Dataset: x in [0,1] -> (sin(2πx-π)+1)/2
    N = 200
    data = []
    for i in range(N):
        t = i / (N - 1)
        y = (math.sin(2 * math.pi * t - math.pi) + 1) / 2
        data.append(([t], [y]))

    for epoch in range(1, 3001):
        random.shuffle(data)
        mse_sum = 0.0
        for x, y in data:
            mse_sum += net.train_step(x, y)
        mse = mse_sum / len(data)
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d}  |  MSE: {mse:.6f}")
        if mse < 2e-5:
            print(f"  Converged at epoch {epoch}  |  MSE: {mse:.6f}")
            break

    # Spot-check
    checks = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\n  {'x':>6}  {'truth':>8}  {'pred':>8}  {'err':>8}")
    for x in checks:
        tru = (math.sin(2 * math.pi * x - math.pi) + 1) / 2
        pred = net.predict([x])[0]
        print(f"  {x:6.2f}  {tru:8.4f}  {pred:8.4f}  {abs(pred-tru):8.4f}")


if __name__ == "__main__":
    demo_xor()
    demo_sine()
    print("\nDone.")
