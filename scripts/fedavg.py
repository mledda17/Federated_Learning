import numpy as np

def fedavg(f_i, x_init, step, num_iter, N_e, N):
    # Initialize variables
    x_i = [x_init.copy() for i in range(N)]
    x_hist = np.zeros((num_iter + 1, x_init.size))
    x_hist[0] = np.mean(x_i, axis=0)

    for k in range(num_iter):
        y_i = []
        for i in range(N):
            w = x_i[i].copy()
            for _ in range(N_e):
                grad = f_i[i].gradient(w)
                w -= step * grad
            y_i.append(w)

        # Coordinator aggregation
        y_k = np.mean(y_i, axis=0)

        # Broadcast new global model
        x_i = [y_k.copy() for _ in range(N)]
        x_hist[k+1] = y_k

    return x_hist