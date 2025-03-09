import torch


@torch.no_grad()
def solve_norm_eqs(
    W: torch.Tensor,
    lmbda: float,
    dtype: torch.dtype,
    device: torch.device,
):
    m, n = W.shape
    G = W @ W.T
    E = torch.eye(m, dtype=dtype, device=device)
    Gi = torch.inverse(G + lmbda * E)

    a = torch.empty(m - 1, m - 1, dtype=dtype, device=device)
    c = torch.empty(m - 1, dtype=dtype, device=device)
    d = torch.empty(1, dtype=dtype, device=device)
    R = torch.empty(m - 1, dtype=dtype, device=device)
    L = torch.empty(m - 1, m - 1, dtype=dtype, device=device)
    M = torch.empty(m - 1, dtype=dtype, device=device)
    Q = torch.zeros((m, m), dtype=dtype, device=device)

    for i in range(m):
        a[:i, :i] = Gi[:i, :i]
        a[:i, i:] = Gi[:i, i + 1 :]
        a[i:, :i] = Gi[i + 1 :, :i]
        a[i:, i:] = Gi[i + 1 :, i + 1 :]
        c[:i] = Gi[i, :i]
        c[i:] = Gi[i, i + 1 :]
        d = Gi[i, i]

        L = a - torch.outer(c / d, c)
        R[:i] = G[:i, i]
        R[i:] = G[i + 1 :, i]
        M = L @ R

        Q[i, :i] = M[:i]
        Q[i, i + 1 :] = M[i:]

    return Q


@torch.no_grad()
def proj_score(W, lmbda=1e-5):
    Q = solve_norm_eqs(W, lmbda, W.dtype, W.device)
    P = Q @ W
    scores = torch.norm(W - P, dim=1)
    return scores
