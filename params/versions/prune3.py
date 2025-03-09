import torch


def fast_prune(
    W: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    amount: int,
    *,
    lmbda: float = 1e-3,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
):
    validate_prune_step(W, b, V, amount)

    device = W.device
    dtype = W.dtype
    n = W.shape[0]

    Q = torch.zeros((n, n), dtype=dtype, device=device)
    M = torch.ones(n, dtype=torch.bool)

    u_alpha = torch.ones(n, dtype=dtype, device=device)
    u_beta = torch.ones(n, dtype=dtype, device=device)
    u_gamma = torch.ones(n, dtype=dtype, device=device)

    G = W @ W.T
    G.diagonal().add_(lmbda)
    Gi = torch.inverse(G)

    for i in range(n):
        mask = torch.arange(n) != i

        A_ = Gi[mask][:, mask]
        b_ = Gi[i, mask]
        d_ = Gi[i, i]
        L = A_ - torch.ger(b_, b_ / d_)

        R = W[mask] @ W[i]
        Q[i, mask] = L @ R

    for _ in range(amount):
        indices = torch.nonzero(M, as_tuple=True)[0]
        dists = torch.norm(Q[M] @ W - W[M], dim=1)
        i = indices[torch.argmin(dists)]

        u_alpha *= 1 + alpha * Q[i]
        u_beta *= 1 + beta * Q[i]
        u_gamma *= 1 + gamma * Q[i]

        M[i] = 0
        W[i] = 0
        Q[:, i] = 0
        Q[M] /= 1 + alpha * Q[i]

    W_ = W[M] * u_alpha[M].unsqueeze(1)
    b_ = b[M] * u_beta[M]
    V_ = V[:, M] * u_gamma[M].unsqueeze(0)

    return W_, b_, V_


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
    assert W.device == b.device == V.device
    assert W.dtype == b.dtype == V.dtype
    assert amount >= 0 and amount < n


def prune(
    linears: list,
    prune_rates: list,
    *,
    lmbda: float = 1e-3,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.5,
):
    sequence = zip(linears[:-1], linears[1:], prune_rates)
    for layer, next_layer, prune_rate in sequence:
        W = layer.weight
        b = layer.bias
        V = next_layer.weight

        amount = int(W.shape[0] * prune_rate)
        W_, b_, V_ = fast_prune(W, b, V, amount, lmbda=lmbda, alpha=alpha, beta=beta, gamma=gamma)

        layer.weight = torch.nn.Parameter(W_)
        layer.bias = torch.nn.Parameter(b_)
        next_layer.weight = torch.nn.Parameter(V_)
