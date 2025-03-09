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
def proj_score(W, lmbda):
    Q = solve_norm_eqs(W, lmbda, W.dtype, W.device)
    P = Q @ W
    scores = torch.norm(W - P, dim=1)
    return scores


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


@torch.no_grad()
def proj_prune(
    Ws: torch.Tensor,
    bs: torch.Tensor,
    Vs: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    lmbda: float,
    amount: int,
    dtype: torch.dtype,
    device: torch.device,
):
    N = len(Ws)
    Ms, Qs, Uas, Ubs, Ugs = [], [], [], [], []
    for i in range(N):
        n = Ws[i].shape[0]
        Ms.append(torch.ones(n, dtype=torch.bool, device=device))
        Qs.append(torch.zeros((n, n), dtype=dtype, device=device))
        Uas.append(torch.ones(n, dtype=dtype, device=device))
        Ubs.append(torch.ones(n, dtype=dtype, device=device))
        Ugs.append(torch.ones(n, dtype=dtype, device=device))
    for i in range(N):
        Qs[i] = solve_norm_eqs(Ws[i], lmbda, dtype=dtype, device=device)

    scores = torch.ones(N)
    for a in range(amount):
        for i in range(N):
            s = torch.norm(Qs[i][Ms[i]] @ Ws[i] - Ws[i][Ms[i]], dim=1)
            s /= torch.sum(s)
            scores[i] = torch.min(s)
        li = torch.argmin(scores)

        r = 1 - a / amount
        Ws[li], Ms[li], Qs[li], Uas[li], Ubs[li], Ugs[li] = proj_prune_step(
            W=Ws[li],
            M=Ms[li],
            Q=Qs[li],
            Ua=Uas[li],
            Ub=Ubs[li],
            Ug=Ugs[li],
            alpha=alpha * r,
            beta=beta * r,
            gamma=gamma * r,
        )

    Ws_ = []
    bs_ = []
    Vs_ = []
    for i in range(N):
        M = Ms[i]
        Ws_.append(Ws[i][M] * Uas[i][M].unsqueeze(1))
        bs_.append(bs[i][M] * Ubs[i][M])
        Vs_.append(Vs[i][:, M] * Ubs[i][M].unsqueeze(0))

    return Ws_, bs_, Vs_
