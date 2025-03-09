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
def solve_self_normal_eqs(
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
):
    device = W.device
    dtype = W.dtype
    n = W.shape[0]
    u_alpha = torch.ones(n, dtype=dtype, device=device)
    u_beta = torch.ones(n, dtype=dtype, device=device)
    u_gamma = torch.ones(n, dtype=dtype, device=device)
    M = torch.ones(n, dtype=torch.bool)
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
):
    validate_prune_step(W, b, V, amount)
    Q = solve_self_normal_eqs(W, lmbda)
    M, ua, ub, ug = proj_prune_iter(W, Q, amount, alpha, beta, gamma)
    W_ = W[M] * ua[M].unsqueeze(1)
    b_ = b[M] * ub[M]
    V_ = V[:, M] * ug[M].unsqueeze(0)
    return W_, b_, V_


@torch.no_grad()
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
        W_, b_, V_ = prune_step(W, b, V, amount, lmbda=lmbda, alpha=alpha, beta=beta, gamma=gamma)
        layer.weight = torch.nn.Parameter(W_)
        layer.bias = torch.nn.Parameter(b_)
        next_layer.weight = torch.nn.Parameter(V_)
