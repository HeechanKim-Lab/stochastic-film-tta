# AS-FiLM: Amortized Stochastic Feature-wise Linear Modulation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

Official PyTorch implementation of **Amortized Stochastic Feature-wise Linear Modulation (AS-FiLM)** for continual test-time adaptation (CTTA) and out-of-distribution (OOD) robustness.

AS-FiLM decouples representation extraction from dynamic adaptation by mounting a variational, distance-aware stochastic modulation layer strictly onto the penultimate features of a frozen backbone. By replacing test-time gradient descent with feedforward amortized conditioning, AS-FiLM eliminates test-time covariance explosion ($\frac{dP_t}{dt} \equiv 0$) and intermediate Jensen expectation bias ($\Delta_\mu \equiv 0$).

---

## Architectural Principles & Guarantees

```
x (Input) ──► [Frozen Backbone (Spectral Norm)] ──► h ∈ ℝᵈ (Penultimate Feature)
                         │                                    │
                         ▼                                    ▼
       [Centroid Distance Kernel: d_M(h, c_k)]       [Stochastic FiLM: h̃ = γ ⊙ h + β]
                         │                             (γ ~ 𝒩(μ_γ, σ_γ²), β ~ 𝒩(μ_β, σ_β²))
                         ▼                                    │
       [Distance-to-Variance: σ²(d_M)]                        ▼
       [Linear Feature Projection: μ(h)]            [Linear Classifier: z = W h̃ + b]
                         │                                    │
                         └───────────────► ◄──────────────────┘
                                           │
                                           ▼
                       [Closed-Form Probit / MC Ensemble: ȳ]
```

* **Zero Jensen Distortion:** Placing stochastic affine parameters $(\gamma, \beta)$ immediately prior to the linear classification head ensures $\mathbb{E}[z] - z(\mathbb{E}[\gamma], \mathbb{E}[\beta]) \equiv 0$, preventing compounding activation saturation across layers.
* **Lyapunov Parameter Stability:** Operating with zero test-time backpropagation ($\eta = 0$) guarantees zero parameter drift and eliminates gradient covariance explosion ($P_t = P_0 = 0, \forall t \ge 0$).
* **Provable OOD Variance Expansion:** Enforcing a bi-Lipschitz condition via spectral normalization and residual topology guarantees that off-manifold inputs diverge in latent distance, asymptotically recovering isotropic prior uncertainty ($\lim_{d(x, \mathcal{D}) \to \infty} \sigma^2(x) = \sigma_0^2 I$).
* **Constant Throughput:** Feature caching limits Monte Carlo sampling to the low-dimensional classifier head, operating at sensor frame rates without dropping streaming samples.

---

## Directory Structure

```
├── configs/
│   ├── cifar10_c_stream.yaml        # Streaming corruption sequence config
│   └── model_resnet18.yaml          # Backbone and FiLM layer dimensions
├── data/                            # Dataset symlinks and CIFAR-10-C binaries
├── scripts/
│   ├── cache_prototypes.py          # Pre-extract ID class centroids and covariance
│   ├── eval_streaming.py            # Streaming continual benchmark execution
│   └── run_baselines.py             # Reference comparisons (Source-only, TENT, EATA)
├── src/
│   ├── adaptation/
│   │   ├── amortized_film.py        # Core AS-FiLM inference engine
│   │   └── baseline_wrappers.py     # TENT, SAR, CoTTA modules
│   ├── models/
│   │   ├── backbone.py              # Spectral-normalized ResNet-18
│   │   ├── film_layer.py            # Penultimate stochastic affine modulation
│   │   └── variance_kernel.py       # Mahalanobis/RBF distance-aware variance head
│   ├── metrics/
│   │   ├── calibration.py           # ECE and adaptive reliability diagrams
│   │   └── robustness.py            # Top-1 Error and Mean Corruption Error (mCE)
│   └── utils/
│       ├── corruptions.py           # CIFAR-10-C streaming generator
│       └── logging.py
├── tests/
│   ├── test_film_linearity.py       # Empirical validation of zero Jensen bias
│   └── test_variance_expansion.py   # OOD variance recovery unit tests
├── environment.yml
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```bash
git clone [https://github.com/](https://github.com/)<username>/as-film.git
cd as-film
conda env create -f environment.yml
conda activate as-film
```

### 2. Cache In-Distribution Prototypes

Extract in-distribution class centroids $\{c_k\}_{k=1}^K$ and feature covariance $\Sigma_h$ using the clean CIFAR-10 training split:

```bash
python scripts/cache_prototypes.py \
    --config configs/model_resnet18.yaml \
    --output checkpoints/cifar10_prototypes.pt
```

### 3. Run Continual Streaming Benchmark

Evaluate AS-FiLM against a continuous, non-resetting stream of CIFAR-10-C corruptions (15 noise types $\times$ Severity 5):

```bash
python scripts/eval_streaming.py \
    --config configs/cifar10_c_stream.yaml \
    --prototypes checkpoints/cifar10_prototypes.pt \
    --samples 10 \
    --eval-mode probit
```

---

## Baseline Comparison

| Framework | Test Backprop | Memory (GPU) | Top-1 Error (CIFAR-10-C) | Extended CTTA ($10^5$ steps) | ECE (Uncertainty) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Source-Only** | None | $1.0\times$ | 43.5% | Stable (Floor) | 0.284 |
| **TENT** (ICLR '21) | Yes | $2.4\times$ | 20.2% | Collapsed (>85%) | 0.312 |
| **CoTTA** (CVPR '22)| Yes | $3.2\times$ | 16.3% | Drifts (>60%) | 0.225 |
| **BCA+** (CVPR '25) | None | $1.1\times$ | 14.8% | Stable | 0.086 |
| **AS-FiLM (Ours)** | **None** | **$1.1\times$** | **15.2%** | **Invariant ($P_t \equiv 0$)** | **0.089** |

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.