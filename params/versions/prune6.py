import math

import torch


@torch.no_grad()
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


@torch.no_grad()
def solve_self_normal_equations(
    W: torch.Tensor,
    lmbda: float,
):
    device = W.device
    dtype = W.dtype
    n = W.shape[0]
    G = W @ W.T
    G.diagonal().add_(lmbda)
    Gi = torch.inverse(G)
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


@torch.no_grad()
def proj_prune_iter(
    W: torch.Tensor,
    Q: torch.Tensor,
    amount: int,
    alpha: float,
    beta: float,
    gamma: float,
    bs: int,
):
    device = W.device
    dtype = W.dtype
    n = W.shape[0]
    u_alpha = torch.ones(n, dtype=dtype, device=device)
    u_beta = torch.ones(n, dtype=dtype, device=device)
    u_gamma = torch.ones(n, dtype=dtype, device=device)
    M = torch.ones(n, dtype=torch.bool, device=device)
    batches = torch.full((math.ceil(amount / bs),), bs, dtype=torch.int64)
    remainder = amount % bs
    if remainder:
        batches[-1] = remainder
    for bs in batches:
        # maybe compute Q here
        indices = torch.nonzero(M, as_tuple=True)[0]
        dists = torch.norm(Q[M] @ W - W[M], dim=1)
        ii = indices[torch.topk(dists, bs, largest=False).indices]
        u_alpha *= torch.prod(1 + alpha * Q[ii], dim=0)
        u_beta *= torch.prod(1 + beta * Q[ii], dim=0)
        u_gamma *= torch.prod(1 + gamma * Q[ii], dim=0)
        M[ii] = 0
        W[ii] = 0
        Q[:, ii] = 0
        Q[M] /= torch.prod(1 + alpha * Q[ii], dim=0)
    return M, u_alpha, u_beta, u_gamma


@torch.no_grad()
def prune_step(
    W: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    amount: int,
    lmbda: float,
    alpha: float,
    beta: float,
    gamma: float,
    bs: int,
):
    validate_prune_step(W, b, V, amount)
    Q = solve_self_normal_equations(W, lmbda)
    M, ua, ub, ug = proj_prune_iter(W, Q, amount, alpha, beta, gamma, bs)
    W_ = W[M] * ua[M].unsqueeze(1)
    b_ = b[M] * ub[M]
    V_ = V[:, M] * ug[M].unsqueeze(0)
    return W_, b_, V_
