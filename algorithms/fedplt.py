import numpy as np
from utilities.quantization import quantize

def fed_plt(f_i, x_init, alpha, num_iter, N_e, N, rho, Delta=None):
    """
    Fed-PLT with uniform quantization on all agent⇄coordinator messages.

    Args:
      f_i      : list of LogisticRegression objects
      x_init   : initial model, shape (n,1)
      alpha    : step size
      num_iter : number of communication rounds
      N_e      : local epochs per round
      N        : number of agents
      rho      : PLT parameter
      Delta    : quantization parameter (None = no quantization)

    Returns:
      x_hist   : array of shape (n,1,num_iter+1) containing the global model at each round
    """
    # Initialize variables
    x_i = [x_init.copy() for _ in range(N)]
    z_i = [x_init.copy() for _ in range(N)]

    x_hist = np.zeros(x_init.shape + (num_iter + 1,))
    x_hist[:, :, 0] = x_init

    for k in range(num_iter):
        #Quantize z_i locally before sending
        z_to_agg = [quantize(z_i[i], Delta) for i in range(N)]
        y_kp1 = sum(z_to_agg) / N

        y_kp1 = quantize(y_kp1, Delta)

        # Local update at each agent
        new_x, new_z = [], []
        for i in range(N):
            w = x_i[i].copy()
            v_i = 2 * y_kp1 - z_i[i]

            for _ in range(N_e):
                grad = f_i[i].gradient(w) + (1 / rho) * (w - v_i)
                w = w - alpha * grad

            x_next = w
            z_next = z_i[i] + 2 * (x_next - y_kp1)

            new_x.append(x_next)
            new_z.append(z_next)

        x_i, z_i = new_x, new_z

        # Record the current aggregated model
        x_hist[:, :, k + 1] = sum(x_i) / N

    return x_hist