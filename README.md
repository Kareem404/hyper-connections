# Manifold-Constrained Hyper-Connections (mHC)
A minimal implementation of mHC proposed by deepseek in January 2026 for attention and MLP layers

## 1. What are Hyper-Connections?
Hyper-Connections are an extension of typical Residual Connections given by:
```math
output= x + F^i(x)
```
where $`F^i`$ is the ith layer.

Hyper Connections extend this idea by adding learnable parameters for n residual streams that act like gates deciding which information to pass from the previous layer (similar to hidden states in LSTMs). Mathematically, a simple static hyper connection is given by:

- $\mathbf{H}^{\text{pre}}_l \in \mathbb{R}^{1 \times n}$: **Pre-mapping** - aggregates $n$ streams into single input