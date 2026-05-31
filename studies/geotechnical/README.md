# Nerve in Geotechnical Engineering — Slope Stability Case Study

Can **Nerve** (the zero-dependency MLP in `../nerve.h`) do useful geotechnical
work? This folder answers that with a real engineering task: predicting the
**Factor of Safety (FS)** of a slope from its soil, geometry, groundwater and
reinforcement parameters.

> FS is *the* number in slope-stability analysis: `FS < 1.0` means failure,
> and most design codes require `FS >= 1.5` for a slope to be considered safe.

## Data

`dataset/slope_stability_dataset.csv` — 10 000 synthetic slope cases.

| Input | Symbol | Unit |
|-------|--------|------|
| Unit weight | γ | kN/m³ |
| Cohesion | c | kPa |
| Internal friction angle | φ | ° |
| Slope angle | β | ° |
| Slope height | H | m |
| Pore-pressure ratio | ru | – |
| Reinforcement type | – | wall / nailing / geosynthetics / drainage (one-hot) |
| **Factor of Safety** | **FS** | **– (target, 0.5–3.0, capped at 3.0)** |

## Method

- **Network:** `10-24-16-1`, Adam + Tanh hidden, Xavier init, L2 = 1e-4.
- Numeric inputs min–max normalised (bounds from the **train split only**);
  reinforcement one-hot encoded (4 columns).
- FS target scaled `[0.5, 3.0] -> [0, 1]` because Nerve's output layer is always
  sigmoid; predictions are scaled back for reporting. The physical cap at FS = 3.0
  sits naturally at sigmoid's upper saturation, which fits this dataset well.
- 80 / 20 train/test split, deterministic seed (42).

## Results (held-out test set, n = 2000)

**A) FS regression**

| Metric | Value |
|--------|-------|
| RMSE | **0.069** FS units |
| MAE | 0.052 FS units |
| R² | **0.989** |
| Baseline RMSE (predict mean FS) | 0.652 |

**B) Safe-slope classification** (threshold FS ≥ 1.5, derived from predicted FS)

| Metric | Value |
|--------|-------|
| Accuracy | **98.2 %** |
| Precision | **100 %** |
| Recall | 97.9 % |

Confusion matrix: **zero false positives** — the model never labels a truly
unsafe slope as safe. Its 37 misses are all *conservative* (safe slopes flagged
as unsafe), which is the right way to err in a safety screen.

## Reproduce

```sh
gcc -O2 slope_stability.c -o slope_stability -lm
./slope_stability            # run from this folder so dataset/ resolves
```

(On Windows / MinGW: `gcc -O2 slope_stability.c -o slope_stability.exe -lm`)

## Takeaway

Nerve learns the slope-stability response surface to within ~0.07 FS units and
screens safe vs. unsafe slopes at 98 % accuracy with no unsafe-as-safe errors —
so it works well as a fast **preliminary FS estimator / design screen**. It does
not replace a limit-equilibrium or FEM analysis for final design, but it is a
solid surrogate for rapid parametric studies, sensitivity sweeps, and early-stage
optioneering. Everything here is self-contained in `geotechnical_tests/`.
