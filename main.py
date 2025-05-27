import numpy as np
import tools
from sklearn.datasets import make_classification
from scripts.fedavg import fed_avg
from scripts.fedplt import fed_plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg as la
import scienceplots

ran = np.random.default_rng()

# Problem parameters
N = 10                   # num. agents
num_data = 100           # per agent
num_features = 5         # size of the model to be trained
reg_weight = 50
step = 0.01
N_e = 10
num_iter = 200
rho = 1.0

# Generate and split data
A, b = make_classification(n_samples=num_data*N, n_features=num_features, n_clusters_per_class=1)

b = b.reshape((-1,1))
b[np.where(b == 0)], b[np.where(b == 1)] = -1, 1

# divide data by agents
A_i = [A[i*num_data:(i+1)*num_data,:] for i in range(N)]
b_i = [b[i*num_data:(i+1)*num_data,:] for i in range(N)]

# Initial Model
ran = np.random.default_rng()
v = ran.normal(size=(num_features, 1))

batch_sizes = [None]

for batch_sz in batch_sizes:
    # Create global cost and local costs
    f = tools.LogisticRegression(A, b,
                                 loss_weight=1/num_data,
                                 reg_weight=50,
                                 batch_sz=batch_sz)
    f_i = [tools.LogisticRegression(A_i[i], b_i[i],
                                    loss_weight=1 / num_data,
                                    reg_weight=reg_weight / N,
                                    batch_sz=batch_sz)
           for i in range(N)]

    # Plot parameters
    pparam = dict(xlabel="Iterations", ylabel="Error")
    N_e_list = [1, 10, 50, 100]

    # FedAvg Plots
    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        fig, ax = plt.subplots()

        for N_e in N_e_list:
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N)
            error = [la.norm(f.gradient(x_avg[..., k])) for k in range(x_avg.shape[-1])]
            ax.semilogy(error, label=fr"${N_e}$")

        ax.legend(title=r"$N_e$")
        ax.autoscale(tight=True)
        ax.set(**pparam)

        fig.savefig("figures/FedAvg/convergence_fedavg_varying_ne.jpg", dpi=300)
        plt.close()

    # FedPLT Plots
    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        fig, ax = plt.subplots()

        for N_e in N_e_list:
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho)
            error = [la.norm(f.gradient(x_plt[..., k])) for k in range(x_avg.shape[-1])]
            ax.semilogy(error, label=fr"${N_e}$")

        ax.legend(title=r"$N_e$")
        ax.autoscale(tight=True)
        ax.set(**pparam)

        fig.savefig("figures/FedPLT/convergence_fedplt_varying_ne.jpg", dpi=300)
        plt.close()