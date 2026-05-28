# Algorithms & Mathematical Reference

This page documents the algorithms implemented in Nerve, with their governing equations and the academic references they are based on.

---

## Forward Propagation

A network of L+1 layers (0 = input, L = output) computes:

**Pre-activation** of neuron j in layer ℓ:

```
z_j^(ℓ) = Σ_i  w_ij^(ℓ) · a_i^(ℓ-1)     (i ranges over neurons + bias unit)
```

**Post-activation:**

```
a_j^(ℓ) = f^(ℓ)( z_j^(ℓ) )
```

The bias unit in every layer has its output permanently set to 1 and contributes to the sum through weight index i = n_(ℓ-1).

---

## Loss Function

Nerve minimises mean squared error (MSE):

```
L(a^(L), t) = ½ Σ_k  (t_k − a_k^(L))²
```

The ½ cancels the exponent in the gradient, yielding a coefficient of −1 on the output error signal.

---

## Backpropagation

*(Rumelhart, Hinton & Williams, 1986)*

**Output-layer delta:**

```
δ_k^(L) = (t_k − a_k^(L)) · f'( z_k^(L) )
```

**Hidden-layer delta (reverse pass):**

```
δ_j^(ℓ) = f'( z_j^(ℓ) ) · Σ_k  w_jk^(ℓ+1) · δ_k^(ℓ+1)
```

**Weight gradient:**

```
∂L / ∂w_ij^(ℓ) = −δ_j^(ℓ) · a_i^(ℓ-1)
```

---

## Activation Functions

The hidden-layer activation is controlled by `net_set_activation()`. The **output layer always uses logistic sigmoid**, regardless of the hidden setting.

| Name | f(z) | f'(a) (via post-activation) |
|------|------|-----------------------------|
| Logistic sigmoid | 1 / (1 + e^−z) | a · (1 − a) |
| Tanh | tanh(z) | 1 − a² |
| ReLU | max(0, z) | 1 if z > 0, else 0 |
| Leaky ReLU | max(0.01z, z) | 1 if z > 0, else 0.01 |

The derivative is always computed from the post-activation value `a`, avoiding a redundant `exp` call (LeCun et al., 1998).

**Choosing an activation:**

- **Tanh**: good default; zero-centred, saturates less than sigmoid.
- **ReLU**: preferred for deeper networks; pair with He initialization.
- **Leaky ReLU**: use when dying-ReLU (zero gradient) is a problem.
- **Sigmoid**: output layer only; occasionally useful as the hidden activation for very shallow networks.

---

## Optimizers

### SGD with Momentum

*(Rumelhart, Hinton & Williams, 1986)*

```
Δw(t)   = η · δ · a  +  μ · Δw(t-1)
w(t+1)  = w(t) + Δw(t)
```

- η = learning rate (default 0.25 for SGD; recommend 0.01–0.1)
- μ = momentum coefficient (default 0.1)

Momentum accumulates velocity in the direction of persistent gradient components, accelerating convergence in ravine-shaped loss surfaces.

### Adam

*(Kingma & Ba, 2015)*

```
m_t = β₁ · m_{t-1}  +  (1 − β₁) · g_t          first moment
v_t = β₂ · v_{t-1}  +  (1 − β₂) · g_t²         second moment

m̂_t = m_t / (1 − β₁ᵗ)                           bias correction
v̂_t = v_t / (1 − β₂ᵗ)

w_{t+1} = w_t  −  η · m̂_t / (√v̂_t + ε)
```

Default hyper-parameters:

| Parameter | Value |
|-----------|-------|
| β₁ | 0.9 |
| β₂ | 0.999 |
| ε | 1e-8 |
| Recommended η | 0.001 – 0.01 |

Adam adapts the learning rate per parameter using estimates of the first and second moments of the gradient. It is robust to sparse gradients and typically requires less learning rate tuning than SGD.

> **Note:** Adam in Nerve is **online-only** (one update per sample). Batch gradient accumulation always uses SGD.

---

## Weight Initialization

### Uniform Random

```
w ~ U[−range, +range]      (default range = 1)
```

The default. No variance scaling; generally inferior to Xavier/He in practice.

### Xavier / Glorot Uniform

*(Glorot & Bengio, 2010)*

```
w ~ U[ −√(6 / (n_in + n_out)),  +√(6 / (n_in + n_out)) ]
```

Derived by requiring the variance of activations and gradients to remain constant across layers (valid for activations that are approximately linear near zero). Recommended with **Sigmoid** and **Tanh**.

### He Uniform

*(He, Zhang, Ren & Sun, 2015)*

```
w ~ U[ −√(6 / n_in),  +√(6 / n_in) ]
```

Corrects Xavier for the ReLU non-linearity, which zeroes half of inputs on average, halving the effective fan-in. Recommended with **ReLU** and **Leaky ReLU**.

---

## Regularization

### L2 Weight Decay

Adds a penalty to the loss proportional to the squared weight norm:

```
L_λ = L  +  λ/2 · Σ w²
```

The gradient of the penalty is added to the weight gradient at update time:

```
g̃ = g  −  λ · w
```

Set with `net_set_l2_lambda(net, lambda)`. Default is 0 (disabled). Typical range: 1e-5 to 1e-3.

### Dropout (Inverted)

*(Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, 2014)*

During each training step, each hidden neuron is independently set to zero with probability p:

```
ã_j = (r_j · a_j) / (1 − p)      r_j ~ Bernoulli(1 − p)
```

The 1/(1-p) scaling (inverted dropout) keeps the expected activation equal to `a_j`, so inference passes require no change. Dropout is applied only to hidden layers during training; the input and output layers are never masked.

Set with `net_set_dropout(net, rate)`. Default is 0 (disabled).

---

## Universal Approximation

The MLP is a universal function approximator: for any continuous function f on a compact domain and any ε > 0, there exists a one-hidden-layer MLP with a finite number of neurons such that the network approximates f to within ε in the sup-norm (Cybenko, 1989; Hornik, Stinchcombe & White, 1989). Nerve's sin(x) example demonstrates this: a 1-16-16-1 network with Tanh achieves MSE < 2×10⁻⁵.

---

## References

1. Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
2. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 249–256.
3. He, K., Zhang, X., Ren, S. & Sun, J. (2015). Delving deep into rectifiers. *ICCV*, 1026–1034.
4. Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
5. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout. *JMLR*, 15, 1929–1958.
6. LeCun, Y., Bottou, L., Orr, G.B. & Müller, K.-R. (1998). Efficient BackProp. *Neural Networks: Tricks of the Trade*, LNCS 1524.
7. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*, 2(4), 303–314.
8. Hornik, K., Stinchcombe, M. & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.
