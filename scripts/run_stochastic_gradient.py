import scienceplots
from utilities import tools
from utilities.save_dat_file import save_dat_file
from algorithms.fedavg import fed_avg
from algorithms.fedplt import fed_plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as la

def run_stochastic_gradient(A, b, A_i, b_i, v, step, num_iter, N, rho, reg_weight):
    batch_sizes = [None, 50, 10, 1]
    N_e = 10
    pparam = dict(xlabel="Iterations", ylabel="Error")

    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        fig_avg, ax_avg = plt.subplots()
        fig_plt, ax_plt = plt.subplots()

        for batch_sz in batch_sizes:
            f = tools.LogisticRegression(A, b, loss_weight=1 / 100, reg_weight=reg_weight, batch_sz=batch_sz)
            f_i = [tools.LogisticRegression(A_i[i], b_i[i], loss_weight=1 / 100, reg_weight=reg_weight / N,
                                            batch_sz=batch_sz) for i in range(N)]
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N)
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho)
            label = "Full batch" if batch_sz is None else f"Batch={batch_sz}"
            err_avg = [la.norm(f.gradient(x_avg[..., k])) for k in range(num_iter + 1)]
            err_plt = [la.norm(f.gradient(x_plt[..., k])) for k in range(num_iter + 1)]
            ax_avg.semilogy(err_avg, label=label)
            ax_plt.semilogy(err_plt, label=label)
            name = "full" if batch_sz is None else f"batch_{batch_sz}"
            save_dat_file(f"figures/FedAvg/fedavg_sgd_{name}.dat", err_avg)
            save_dat_file(f"figures/FedPLT/fedplt_sgd_{name}.dat", err_plt)

        ax_avg.legend(title="Batch size")
        ax_avg.set(**pparam)
        ax_avg.set_title("FedAvg with Stochastic Gradients")
        fig_avg.savefig("figures/FedAvg/convergence_fedavg_sgd.jpg", dpi=300)
        plt.close(fig_avg)

        ax_plt.legend(title="Batch size")
        ax_plt.set(**pparam)
        ax_plt.set_title("Fed-PLT with Stochastic Gradients")
        fig_plt.savefig("figures/FedPLT/convergence_fedplt_sgd.jpg", dpi=300)
        plt.close(fig_plt)