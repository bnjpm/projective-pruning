import torch


def prune_projective(
        W: torch.Tensor,  # shape (n, m)
        b: torch.Tensor,  # shape (n,)
        V: torch.Tensor,  # shape (k, n)
        *,
        lmbda: float = 1e-3,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.5,
):
    device = W.device
    n = W.shape[0]

    # precompute Gram matrix
    G = W @ W.T                                  # shape (n, n)
    L = torch.eye(n-1).to(device) * lmbda        # shape (n-1, n-1)
    coeffs = torch.empty(n, n-1).to(device)      # shape (n, n-1,)
    distances = torch.empty(n)                   # shape (n,)

    for i in range(n):
        wi = W[i]                                # shape (m,)
        minus_i_mask = torch.arange(n) != i      # shape (n,)
        Wmi = W[minus_i_mask]                    # shape (n-1, m)
        Gmi = G[minus_i_mask][:, minus_i_mask]   # shape (n-1, n-1)

        # projecting wi
        qhat = torch.inverse(Gmi+L) @ Wmi @ wi   # shape (n-1,)
        proj = Wmi.T @ qhat                      # shape (m,)
        dist = torch.norm(wi - proj)             # scalar

        coeffs[i] = qhat
        distances[i] = dist

    # prune p-th row
    p = torch.argmin(distances)
    minus_p_mask = torch.arange(n) != p          # shape (n,)
    W_ = W[minus_p_mask]                         # shape (n-1, m)
    b_ = b[minus_p_mask]                         # shape (n-1,)
    V_ = V[:, minus_p_mask]                      # shape (k, m-1)

    # update weights (using broadcasting)
    qhat = coeffs[p]                             # shape (n-1,)
    W_.add_(W_ * alpha * qhat.unsqueeze(1))      # shape (n-1, m)
    b_.add_(b_ * beta * qhat)                    # shape (n-1,)
    V_.add_(V_ * gamma * qhat.unsqueeze(0))      # shape (k, n-1)

    return W_, b_, V_


def validate_prune_input(
        A: torch.Tensor,  # shape (n, m)
        a: torch.Tensor,  # shape (n,)
        B: torch.Tensor,  # shape (k, n)
):
    m = A.shape[1]
    n = A.shape[0]
    k = B.shape[0]

    assert A.shape == (n, m)
    assert a.shape == (n,)
    assert B.shape == (k, n)

    assert A.device == a.device == B.device


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

        A = layer.weight
        a = layer.bias
        B = next_layer.weight
        validate_prune_input(A, a, B)

        prune_amount = int(A.shape[0] * prune_rate)
        for _ in range(prune_amount):

            A, a, B = prune_projective(
                A, a, B,
                lmbda=lmbda,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            layer.weight = torch.nn.Parameter(A)
            layer.bias = torch.nn.Parameter(a)
            next_layer.weight = torch.nn.Parameter(B)
