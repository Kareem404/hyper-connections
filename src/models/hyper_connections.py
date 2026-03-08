import torch
import torch.nn as nn


class mHC_attn(nn.Module):
    """
    Implementation of Manifold-Constrained Hyper connections for an attention layer
        
    args:-
    - layer(nn.Module): An attention layer with input shape: [b, T, d_model]
    - expansion_rate(int): Expansion rate (i.e. how many copies of the input d_model should be created)
    - d(int): The embedding dimension of the attention layer
    - T(int): The maximum sequence length or context for attention 
    """

    def __init__(
        self, 
        layer: nn.Module, 
        expansion_rate: int, 
        d: int, 
        T: int
    ):
        super().__init__()

        self.n = expansion_rate
        self.layer = layer
        self.d = d
        self.T = T

        # init dyanmic parameters (linear projections) decided by input shape
        self.theta_pre = nn.Parameter(torch.zeros(self.n * self.d, self.n), requires_grad=True) # [n*d, n]
        self.theta_post = nn.Parameter(torch.zeros(self.n * self.d, self.n), requires_grad=True) # [n*d, n]
        self.theta_res = nn.Parameter(torch.zeros(self.n * self.d, self.n*self.n), requires_grad=True) # [n*d, n*n]

        # init alpha (learnable gating factors) by a small value
        self.alpha_pre = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.alpha_post = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.alpha_res = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # init static mappings (learnable biases)
        self.b_pre = nn.Parameter(torch.ones(1, self.n) / self.n, requires_grad=True) # [1, n]
        self.b_post = nn.Parameter(torch.ones(1, self.n), requires_grad=True) # [1, n] 
        self.b_res = nn.Parameter(torch.eye(self.n), requires_grad=True) # [n, n]

        self.rmsnorm = nn.RMSNorm(normalized_shape=(self.n*d,))
    
    def sinkhorn_knopp(self, matrix: torch.tensor, t: int=20):
        """
        Applies the sinkhorn-knopp algorithm for an nxn matrix
        args:-
            matrix(torch.tensor): a matrix or a batch of matrecies as input
            t(int): number of iterations (paper suggests 20 iterations)
        """
        assert type(t) == int

        m = torch.exp(matrix)
        for _ in range(t):
            m = m / m.sum(dim=-1, keepdim=True)

            m = m / m.sum(dim=-2, keepdim=True)
            
        return m
    
    def forward(self, x):
        """
        forward pass through mHC attention
        args:-
        - x: x.shape = [b, T, n, d]
        """

        b, T, n, d = x.shape
        x = x.view(b*T, n, d)
        x = x.view(b*T, 1, n*d) # [b*T, 1, n*d]

        x_norm = self.rmsnorm(x) # [b*T, 1, n*d] 

        H_pre = torch.sigmoid(self.alpha_pre * (x_norm @ self.theta_pre) + self.b_pre) # [b*T, 1, n]

        H_post = 2 * torch.sigmoid(self.alpha_post * (x_norm @ self.theta_post) + self.b_post).squeeze(1) # [b*T, n]

        H_res = self.alpha_res * (x_norm @ self.theta_res).view(b*T, n, n) + self.b_res # [b*T, n, n]

        # apply sinkhorn-knopp
        H_res = self.sinkhorn_knopp(H_res) # [b*T, n, n]

        x = x.view(b*T, n, d) # [b*T, n, d]

        x_pre = (H_pre @  x).squeeze(1) # [b*T, d]

        x_pre = x_pre.view(b, T, d) # [b, T, d]

        H_bar = self.layer(x_pre) # [b, T, d] 

        H_bar = H_bar.view(b*T, d) # [b*T, d]

        z_l = torch.einsum("bn,bd->bnd", H_post, H_bar) # [b*T, n, d]

        h_l = H_res @ x # [b*T, n, d]

        return z_l + h_l # [b*T, n, d]
    

class mHC_mlp(nn.Module):
    """
    Class that applies mHC for a multi-layer preceptron of shape [b, n, d] where n is the expansion rate

    args:-
    - layer(nn.Module): an MLP layer where input shape = output shape 
    - expansion_rate(int): Expansion rate (i.e. how many copies of the input d_model should be created)
    - d(int): The embedding dimension of the attention layer
    """

    def __init__(self, 
        layer: nn.Module, 
        expansion_rate: int, 
        d: int
    ):

        super().__init__()

        self.n = expansion_rate
        self.layer = layer
        self.d = d

        # init dyanmic parameters (linear projections) decided by input shape
        self.theta_pre = nn.Parameter(torch.zeros(self.n * self.d, self.n), requires_grad=True) # [n*d, n]
        self.theta_post = nn.Parameter(torch.zeros(self.n * self.d, self.n), requires_grad=True) # [n*d, n]
        self.theta_res = nn.Parameter(torch.zeros(self.n * self.d, self.n*self.n), requires_grad=True) # [n*d, n*n]

        # init alpha (learnable gating factors) by a small value
        self.alpha_pre = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.alpha_post = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.alpha_res = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # init static mappings (leanable biases)
        self.b_pre = nn.Parameter(torch.ones(1, self.n) / self.n, requires_grad=True) # [1, n]
        self.b_post = nn.Parameter(torch.ones(1, self.n), requires_grad=True) # [1, n] 
        self.b_res = nn.Parameter(torch.eye(self.n), requires_grad=True) # [n, n]

        self.rmsnorm = nn.RMSNorm(normalized_shape=(self.n*d,))
    
    def sinkhorn_knopp(self, matrix, t=20):
        """
        Applies the sinkhorn-knopp algorithm for an nxn matrix
        args:-
        - matrix(torch.tensor): a matrix or a batch of matrecies as input
        - t(int): number of iterations (paper suggests 20 iterations)
        """
        m = torch.exp(matrix)
        for _ in range(t):
            m = m / m.sum(dim=-1, keepdim=True)

            m = m / m.sum(dim=-2, keepdim=True)
            
        return m
    
    def forward(self, x):
        """
        args:-
            x: x.shape = [b, n, d]
        """
        b, n, d = x.shape
        x = x.view(b, 1, n*d) # [b, 1, n*d]

        x_norm = self.rmsnorm(x) # [b, 1, n*d] 

        H_pre = torch.sigmoid(self.alpha_pre * (x_norm @ self.theta_pre) + self.b_pre) # [b, 1, n]

        H_post = 2 * torch.sigmoid(self.alpha_post * (x_norm @ self.theta_post) + self.b_post).squeeze(1) # [b, n]

        H_res = self.alpha_res * (x_norm @ self.theta_res).view(b, n, n) + self.b_res # [b, n, n]

        # apply sinkhorn-knopp
        H_res = self.sinkhorn_knopp(H_res) # [b, n, n]

        x = x.view(b, n, d)

        x_pre = (H_pre @  x).squeeze(1) # [b, d]

        H_bar = self.layer(x_pre) # [b, d]

        z_l = torch.einsum("bn,bd->bnd", H_post, H_bar) # [b, n, d]

        h_l = H_res @ x # [b, n, d]

        return z_l + h_l # [b, n, d]