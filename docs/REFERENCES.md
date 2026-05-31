# Nerve — academic foundations

Nerve is engineering, not new science: it is a from-scratch, zero-dependency,
single-header implementation of well-established results, written to be read.
Every component below is grounded in its original paper. Nothing here is
hand-waved — the code is the proof, and these are the references it rests on.

The contribution is not a new algorithm; it is *readability, portability and
reproducibility* of the modern stack in pure C — and demonstrating, with
runnable artifacts, that the whole pipeline (train → embed → generate → learn →
deploy) fits in code a person can hold in their head.

---

## Training the MLP core (`nerve.h`)

| Component | Reference |
|-----------|-----------|
| Backpropagation | Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors*, Nature, 1986 |
| Adam optimizer | Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015 |
| Xavier/Glorot init | Glorot & Bengio, *Understanding the difficulty of training deep feedforward networks*, AISTATS 2010 |
| He init | He, Zhang, Ren & Sun, *Delving Deep into Rectifiers*, ICCV 2015 |
| ReLU | Nair & Hinton, 2010; Glorot, Bordes & Bengio, 2011 |
| Dropout | Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014 |
| Softmax + cross-entropy | Bridle, *Probabilistic interpretation of feedforward classification network outputs*, 1990 |

## Automatic differentiation (`studies/autograd/nerve_grad.h`)

| Component | Reference |
|-----------|-----------|
| Reverse-mode autodiff | Linnainmaa, 1970; Griewank & Walther, *Evaluating Derivatives*, 2008 |
| Modern survey | Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*, JMLR 2018 |
| Minimal didactic engines | micrograd (Karpathy); tinygrad — prior art for "small but real" autodiff |

## Decoder Transformer inference (`studies/infer/nerve_infer.h`)

| Component | Reference |
|-----------|-----------|
| Transformer / scaled dot-product & multi-head attention | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 |
| Rotary position embedding (RoPE) | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 |
| RMSNorm | Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019 |
| SwiGLU feed-forward | Shazeer, *GLU Variants Improve Transformer*, 2020 |
| SiLU / Swish activation | Elfwing et al., 2018; Ramachandran, Zoph & Le, 2017 |
| Grouped-query attention | Ainslie et al., *GQA*, 2023 |
| Llama / Llama 2 architecture | Touvron et al., *LLaMA*, 2023; *Llama 2*, 2023 |
| Nucleus (top-p) sampling | Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 |
| TinyStories (small-model coherence) | Eldan & Li, *TinyStories*, 2023 |
| Single-file inference, prior art | Karpathy, *llama2.c* (architecture/format reference) |

## Sentence-embedding encoder (`studies/embed/nerve_embed.h`)

| Component | Reference |
|-----------|-----------|
| BERT bidirectional encoder | Devlin et al., *BERT*, NAACL 2019 |
| Layer Normalization | Ba, Kiros & Hinton, *Layer Normalization*, 2016 |
| GELU activation | Hendrycks & Gimpel, *Gaussian Error Linear Units*, 2016 |
| WordPiece tokenization | Schuster & Nakajima, 2012; Wu et al., *GNMT*, 2016 |
| Sentence embeddings (mean-pool) | Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019 |
| MiniLM distillation | Wang et al., *MiniLM: Deep Self-Attention Distillation*, NeurIPS 2020 |
| Embedding anisotropy fix (mean-centering) | Mu & Viswanath, *All-but-the-Top*, ICLR 2018 |

## Efficiency

| Component | Reference |
|-----------|-----------|
| int8 quantization | Jacob et al., *Quantization and Training … Integer-Arithmetic-Only Inference*, CVPR 2018 |
| LLM int8 context | Dettmers et al., *LLM.int8()*, NeurIPS 2022 |

## On-device & decentralized learning (direction)

| Component | Reference |
|-----------|-----------|
| Low-rank adaptation (LoRA) | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 |
| Federated learning | McMahan et al., *Communication-Efficient Learning … from Decentralized Data*, AISTATS 2017 |
| Low-communication distributed training | Douillard et al., *DiLoCo*, 2023 |
| Decentralized LLM training (existence proof) | Prime Intellect, *INTELLECT-1*, 2024 |

---

*If you find a result implemented here without its citation, that is a bug —
please open an issue.*
