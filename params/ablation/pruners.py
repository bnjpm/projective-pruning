import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.linear import Linear


def ln_prune(
    first: Linear,
    second: Linear,
    amount: int,
    norm: int = 2,
):
    W = first.weight
    b = first.bias
    V = second.weight
    n = W.shape[0]

    norms = torch.norm(W, p=norm, dim=1)
    mask = torch.topk(norms, n - amount, largest=True).indices
    W_ = W[mask]
    b_ = b[mask]
    V_ = V[:, mask]

    first.weight = nn.Parameter(W_)
    first.bias = nn.Parameter(b_)
    second.weight = nn.Parameter(V_)

    return first, second


def validate_prune_step(
    W: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    amount: int,
):
    m = W.shape[1]
    n = W.shape[0]
    k = V.shape[0]
    assert W.shape == (n, m)
    assert b.shape == (n,)
    assert V.shape == (k, n)
    assert W.dtype == b.dtype == V.dtype
    assert W.device == b.device == V.device
    assert amount > 0 and amount < n


def solve_self_normal_equations(
    W: torch.Tensor,
    lmbda: float,
    dtype: torch.dtype,
    device: torch.device,
):
    n = W.shape[0]
    G = W @ W.T
    E = torch.eye(n, dtype=dtype, device=device)
    Gi = torch.inverse(G + lmbda * E)
    Q = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        mask = torch.arange(n) != i
        a = Gi[mask][:, mask]
        c = Gi[i, mask]
        d = Gi[i, i]
        L = a - torch.ger(c / d, c)
        R = W[mask] @ W[i]
        Q[i, mask] = L @ R
    return Q


def proj_prune_iter(
    W: torch.Tensor,
    amount: int,
    alpha: float,
    beta: float,
    gamma: float,
    lmbda: float,
    dtype: torch.dtype,
    device: torch.device,
):
    n = W.shape[0]
    M = torch.ones(n, dtype=torch.bool, device=device)
    Q = solve_self_normal_equations(W, lmbda, dtype, device)
    Ua = torch.ones(n, dtype=dtype, device=device)
    Ub = torch.ones(n, dtype=dtype, device=device)
    Ug = torch.ones(n, dtype=dtype, device=device)

    for _ in range(amount):
        indices = torch.nonzero(M, as_tuple=True)[0]
        dists = torch.norm(Q[M] @ W - W[M], dim=1)
        i = indices[torch.argmin(dists)]
        M[i] = 0
        W[i] = 0
        Q[:, i] = 0
        Q[M] /= 1 + alpha * Q[i]
        Ua *= 1 + alpha * Q[i]
        Ub *= 1 + beta * Q[i]
        Ug *= 1 + gamma * Q[i]
    return M, Ua, Ub, Ug


def proj_prune(
    W: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    amount: int,
    alpha: float,
    beta: float,
    gamma: float,
    lmbda: float,
):
    validate_prune_step(W, b, V, amount)
    M, Ua, Ub, Ug = proj_prune_iter(W, amount, alpha, beta, gamma, lmbda, W.dtype, W.device)
    W_ = W[M] * Ua[M].unsqueeze(1)
    b_ = b[M] * Ub[M]
    V_ = V[:, M] * Ug[M].unsqueeze(0)
    return W_, b_, V_


def proj_prune_conv(
    c0: nn.Conv2d,
    bn: nn.BatchNorm2d,
    c1: nn.Conv2d,
    amount: int,
    alpha: float,
    beta: float,
    gamma: float,
    lmbda: float,
):
    out0, in0, h0, w0 = c0.weight.shape
    W = rearrange(c0.weight, "out_c in_c h w -> out_c (in_c h w)")
    b = c0.bias

    out1, in1, h1, w1 = c1.weight.shape
    V = rearrange(c1.weight, "out_c in_c h w -> in_c (out_c h w)")

    validate_prune_step(W, b, V.T, amount)
    M, Ua, Ub, Ug = proj_prune_iter(W, amount, alpha, beta, gamma, lmbda, W.dtype, W.device)
    W_ = W[M] * Ua[M].unsqueeze(1)
    b_ = b[M] * Ub[M]
    V_ = V[M] * Ug[M].unsqueeze(1)

    W_ = rearrange(
        W_,
        "out_c (in_c h w) ->  out_c in_c h w",
        out_c=out0 - amount,
        in_c=in0,
        h=h0,
        w=w0,
    )
    V_ = rearrange(
        V_,
        "in_c (out_c h w) ->  out_c in_c h w",
        out_c=out1,
        in_c=in1 - amount,
        h=h1,
        w=w1,
    )

    c0.weight.data = W_
    c0.bias.data = b_
    c1.weight.data = V_

    bn.weight.data = bn.weight[M]
    bn.bias.data = bn.bias[M]
    bn.running_mean.data = bn.running_mean[M]
    bn.running_var.data = bn.running_var[M]

    return c0, bn, c1
