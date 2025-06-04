import numpy as np
from utilities.quantization import quantize

def fed_avg(f_i, x_init, alpha, num_iter, N_e, N, Delta=None):
    """
    FedAvg with uniform quantization on the agent -> coordinator and coordinator -> agent links.

    Args:
        f_i: list of LogisticRegression objects.
        x_init: (n, 1) initial model
        alpha: step size
        num_iter: number of communication rounds
        N_e: local epochs for round
        N: number of agents
        Delta: quantization parameter (None = no quantization)

    Returns:
        x_hist of shape(n, 1, num_iter + 1)
    """
    # Initialize variables
    x_i = [x_init.copy() for _ in range(N)]
    x_hist = np.zeros(x_init.shape + (num_iter + 1,))
    x_hist[:, :, 0] = x_init

    for k in range(num_iter):
        # Local -> Quantize -> Aggregate
        y_i = []
        for i in range(N):
            w = x_i[i].copy()
            for _ in range(N_e):
                w -= alpha * f_i[i].gradient(w)
            y_i.append(quantize(w, Delta))

        # Coordinator aggregation
        y_k = sum(y_i) / N
        y_k = quantize(y_k, Delta)

        # Broadcast new global model
        x_i = [y_k.copy() for _ in range(N)]
        x_hist[:, :, k+1] = y_k

    return x_hist