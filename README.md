# Manifold-Constrained Hyper-Connections (mHC)
A minimal implementation of mHC proposed by deepseek in January 2026 for attention and MLP layers tested with GPT. 

## 1. What are Hyper-Connections?
Hyper-Connections are an extension of typical Residual Connections given by:

$$x_{l+1} = x_l + F_l(x_l)$$

where $`F_l`$ is the $`l-th`$ layer.

### Intuition

Instead of a single residual pathway, Hyper-Connections maintain **multiple parallel residual streams**.

Each layer dynamically decides:
- how much information to aggregate from the streams
- how to redistribute the transformed representation

This allows richer information flow compared to standard residual connections.

Hyper-Connections extend the idea of residual connections by adding learnable parameters for $n$ residual streams that act like gates deciding which information to pass from the previous layer (similar to hidden states in LSTMs). Mathematically, a simple static hyper  connection that does not depend on the input is:

Given input $\mathbf{x}_l \in \mathbb{R}^{n \times d}$ at layer $l$ where $n$ is the expansion rate and $d$ is input dimension:

$$x_l^{pre} = H_l^{pre}x_l \in \mathbb{R}^{1 \times d}$$
$$z_l = H_l^{post}F^i(x_l^{pre}) \in \mathbb{R}^{n \times d}$$
$$h_l = H_l^{res}x_l \in \mathbb{R}^{n \times d}$$
$$x_{l+1}=h_l+z_l$$

where $H_l^{pre} \in \mathbb{R}^{1 \times n}$, $H_l^{post} \in \mathbb{R}^{1 \times n}$, and $H_l^{res} \in \mathbb{R}^{n \times n}$ are learnable parameters

In case we wanted $H_l^{pre}$, $H_l^{post}$, and $H_l^{res}$ to depend on the input, we can rewrite them as follows:

$$x_l = RMSNORM(x_l)$$
$$H_l^{pre} = \alpha_l^{pre} tanh(\theta_l^{pre} x_l^T) +b_l^{pre}$$
$$H_l^{post} = \alpha_l^{post} tanh(\theta_l^{post} x_l^T) +b_l^{post}$$
$$H_l^{res} = \alpha_l^{res} tanh(\theta_l^{res} x_l^T) +b_l^{res}$$

where $\alpha_l^{pre}$, $\alpha_l^{post}$, $\alpha_l^{res}$, $b_l^{pre}$, $b_l^{post}$, and $b_l^{res}$ $\in \mathbb{R}$ are learnable parameters.

This approach is called Dyanmic Hyper Connections (DHC). In Dynamic Hyper-Connections (DHC), the routing matrices depend on the input representation, allowing the model to dynamically adapt information flow across layers.


Deepseek builds on top of this idea to make DHCs more stable by making $H_l^{res}$ doubly stochastic using the sinkhorn-knopp algorithm and by applying a sigmoid activation for both $H_l^{pre}$ and $H_l^{post}$. We can represent them by:

$$x_l = reshape(x_l)$$

where $x_l \in \mathbb{R}^{1 \times n*d}$

$$H_l^{pre} = \sigma(\alpha_l^{pre} (\theta_l^{pre} x_l^T) +b_l^{pre})$$
$$H_l^{post} = 2 \times \sigma(\alpha_l^{post} (\theta_l^{post} x_l^T) +b_l^{post})$$
$$H_l^{res} = \text{sinkhorn-knopp}(\alpha_l^{res} reshape(\theta_l^{res} x_l^T) +b_l^{res})$$

where $\theta_l^{res} x_l^T \in \mathbb{R}^{n \times n}$

## 2. Usage

### 2.1 Installation
Clone the repo:
```
git clone https://github.com/Kareem404/hyper-connections.git
```

Make a virtual environment:
```
python -m venv .venv
```

Activate the environment (Windows) by running:
```
./.venv/Scripts/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

### 2.2 Running the Model
To run a GPT model:
- Install a text corpus like Tiny Shakespeare from this [link](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and save it as a `.txt` file. 

- Modify the model's hyperparameters in the `./configs/[model].yaml`. Make sure to have `hc: true` to use mHC.

Run the following command:
```
python .\scripts\train_gpt.py \
--experiment [experiment_name] \
--dataset [dataset_path]
```
The model and plots will be saved in the `./results` directory.

### 2.3 Wrapping your own layers with mHC
If you want to wrap your own attention or MLP layers with mHC, import `mhc_attn` or `mhc_mlp` from `./src/models/hyper_connections.py` and pass your attention or MLP layers to them.

For `mhc_attn`, the input tensor must include the **stream dimension `n`**.

Use this code snippet for reference:-
```
class TransformerBlock():
    def __init__()
        ...
        self.attn = MultiHeadAttention(
            num_heads=num_heads, 
            d_model=self.d_model,
            dropout=dropout
        )
        self.mhc_attn = mHC_attn(
            self.attn, 
            expansion_rate=self.n, 
            d=self.d_model, 
            T=self.T
        )
        ...
    def forward(H):
        # H.shape = [b, T, n, d]
        H = self.mhc_attn(H) # [b*T, n, d]
        H = H.view(b, T, n, d)
```

After applying multiple transformer layers, you can sum over `n` to make the shape `[b, T, d_model]`:
```
H = H.sum(dim=2)
```

## 3. Results
 
A baseline GPT model with standard residual connections was compared against GPT using mHC.

| Model | Validation Loss | Perplexity |
|------|------|------|
| GPT (Residual Connections) | ~1.57 | ~4.89 |
| GPT + mHC | ~1.58–1.60 | ~4.9 |

During training, the mHC model converged slower than the baseline GPT but reached a similar final validation loss. Gradient norms was unstable at the beginning of the training for mHC but stabalized later in the training.

Because the experiment uses a small model, character tokenization, and a simple dataset, the benefits of mHC are not clearly visible in this setting. More complex datasets or larger models may better demonstrate the advantages of hyper-connections.

More experiements will be conducted soon with ViTs, different hyperparameters and different datasets.

## 4. References

1. **Zhu et al., 2025.** *Hyper-Connections*. ICLR 2025.  
   https://arxiv.org/abs/2409.19606

2. **Xie et al., 2026.** *mHC: Manifold-Constrained Hyper-Connections*.  
   https://arxiv.org/abs/2512.24880

## 5. Additional Resources

- How Residual Connections Are Getting an Upgrade (mHC)  
  https://youtu.be/jYn_1PpRzxI
