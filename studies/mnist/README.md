# MNIST — the "is it real?" benchmark

MNIST is machine learning's universal yardstick: 70 000 handwritten digits
(28×28 grayscale, 60 000 train / 10 000 test). Every framework is measured
against it. So we point **Nerve** straight at it — no convolutions, no BLAS,
no NumPy, no Python — just a plain fully-connected network compiled from one
header file, learning from raw pixels.

## Method

- **Inputs:** 784 raw pixels, scaled to `[0, 1]`.
- **Network:** `784-128-10`, Adam optimizer, ReLU hidden (He init).
  Output layer is sigmoid (Nerve's default); the predicted digit is the
  argmax over the 10 outputs.
- **Targets:** one-hot 10-vectors.
- **Training:** stochastic (one Adam step per image), shuffled each epoch.
- Deterministic seed (42) for reproducibility.

No feature engineering, no augmentation, no tricks — the point is to show the
core library learns a real, non-trivial dataset.

## Results

**96.97 % test accuracy** on the 10 000-image hold-out set — a plain
fully-connected network, raw pixels, no convolutions, written in one C header.

| epoch | train MSE | test accuracy |
|------:|----------:|--------------:|
| 1 | 0.05456 | 96.18 % |
| 2 | 0.03547 | 96.39 % |
| 3 | 0.03193 | 96.90 % |
| 4 | 0.02992 | 96.13 % |
| 5 | 0.02900 | 96.56 % |
| 6 | 0.02879 | **96.97 %** |

Per-digit accuracy (test set):

| digit | acc | | digit | acc |
|:-----:|----:|---|:-----:|----:|
| 0 | 98.16 % | | 5 | 94.06 % |
| 1 | 98.77 % | | 6 | 97.08 % |
| 2 | 96.41 % | | 7 | 97.47 % |
| 3 | 97.82 % | | 8 | 98.05 % |
| 4 | 97.15 % | | 9 | 94.25 % |

The two weakest digits — **5** and **9** — are the classic MNIST confusions
(5↔6/8, 9↔4/7), exactly where humans and big models stumble too. With a
learning-rate schedule and more epochs an MLP reaches ~98 %; we stop at a clean,
honest 6-epoch run on a normal CPU.



## Reproduce

```sh
# 1. Fetch the dataset (~11 MB, IDX files into ./data)
sh download_data.sh            # or:  ./download_data.ps1   on Windows

# 2. Build
gcc -O2 mnist.c -o mnist -lm   # or:  make

# 3. Train + evaluate (args: epochs, train_cap; defaults 10 60000)
./mnist                        # or:  make run
./mnist 6 60000                # what produced the numbers above
./mnist 2 10000                # quick taste: ~92% after one 10k epoch
```

The dataset itself is **not** committed (large binaries); the download scripts
pull it from the public OSSCI S3 mirror of the original LeCun files.

## Notes & honest caveats

- A fully-connected MLP tops out around **97–98 %** on MNIST; the remaining gap
  to state-of-the-art (~99.7 %) needs convolutions, which Nerve does not have.
  That ceiling is the point: we are showing the library is *real and correct*,
  not chasing a leaderboard.
- Training is single-threaded scalar C. It is not fast, but it is honest — every
  weight update is plain, readable code in `../../nerve.h`. The headline is
  **"a neural net you can read end-to-end, with zero dependencies, learns MNIST."**
