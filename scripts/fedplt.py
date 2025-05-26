import numpy as np

def fedplt(f_i, x_init, step, num_iter, N_e, N, rho):
    # Initialize variables
    x_i = [x_init.copy() for _ in range(N)]
    z_i = [x_init.copy() for _ in range(N)]
    x_hist = np.zeros((num_iter + 1, x_init.size))
    x_hist[0] = np.mean(x_i, axis=0)

    for k in range(num_iter):
        # Coordinator aggregation step
        y_kp1 = np.mean(z_i, axis=0)

        # Local update at each agent
        for i in range(N):
            w = x_i[i].copy()
            v_i = 2 * y_kp1 - z_i[i]

            for _ in range(N_e):
                grad = f_i[i].gradient(w) + (1 / rho) *(w - v_i)
                w -= step * grad

            x_i[i] = w.copy()

            # Update auxiliary variable
            z_i[i] += 2 * (x_i[i] - y_kp1)

        # Record current aggregated model
        x_hist[k + 1] = np.mean(x_i, axis=0)

    return x_hist