import scienceplots
from utilities.save_dat_file import save_dat_file
from algorithms.fedavg import fed_avg
from algorithms.fedplt import fed_plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg as la


def comparison_varying_ne(f_i, f, v, step, num_iter, N, rho):
    N_e_list = [1, 10, 50, 100]
    pparam = dict(xlabel="Iterations", ylabel="Error")

    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        # FedAvg
        fig, ax = plt.subplots()
        for N_e in N_e_list:
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N)
            error = [la.norm(f.gradient(x_avg[..., k])) for k in range(x_avg.shape[-1])]
            ax.semilogy(error, label=fr"$N_e = {N_e}$")
            save_dat_file(f"figures/FedAvg/fedavg_ne_{N_e}.dat", error)
        ax.legend()
        ax.set(**pparam)
        fig.savefig("figures/FedAvg/convergence_fedavg_varying_ne.pdf", dpi=300)
        plt.close(fig)

        # FedPLT
        fig, ax = plt.subplots()
        for N_e in N_e_list:
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho)
            error = [la.norm(f.gradient(x_plt[..., k])) for k in range(x_plt.shape[-1])]
            ax.semilogy(error, label=fr"$N_e = {N_e}$")
            save_dat_file(f"figures/FedPLT/fedplt_ne_{N_e}.dat", error)
        ax.legend()
        ax.set(**pparam)
        fig.savefig("figures/FedPLT/convergence_fedplt_varying_ne.pdf", dpi=300)
        plt.close(fig)
