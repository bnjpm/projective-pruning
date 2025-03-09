import torch


def solve_norm_eqs(
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


@torch.no_grad()
def proj_prune_step(
    W: torch.Tensor,
    M: torch.Tensor,
    Q: torch.Tensor,
    Ua: torch.Tensor,
    Ub: torch.Tensor,
    Ug: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
):
    indices = torch.nonzero(M, as_tuple=True)[0]
    dists = torch.norm(Q[M] @ W - W[M], dim=1)
    i = indices[torch.argmin(dists)]
    W[i] = 0
    M[i] = 0
    Q[:, i] = 0
    Q[M] /= 1 + alpha * Q[i]
    Ua *= 1 + alpha * Q[i]
    Ub *= 1 + beta * Q[i]
    Ug *= 1 + gamma * Q[i]
    return W, M, Q, Ua, Ub, Ug
