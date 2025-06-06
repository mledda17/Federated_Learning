import scienceplots
from utilities.save_dat_file import save_dat_file
from algorithms.fedavg import fed_avg
from algorithms.fedplt import fed_plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg as la

def run_quantization(f_i, f, v, step, num_iter, N, rho):
    Delta_list = [None, 1e-3, 1e-2, 1e-1, 1.0]
    N_e = 10
    pparam = dict(xlabel="Iterations", ylabel="Error")

    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        # FedAvg
        fig, ax = plt.subplots()
        for Delta in Delta_list:
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N, Delta=Delta)
            error = [la.norm(f.gradient(x_avg[..., k])) for k in range(x_avg.shape[-1])]
            label = "No quant." if Delta is None else fr"$\Delta={Delta}$"
            ax.semilogy(error, label=label)
            name = "noquant" if Delta is None else f"delta_{Delta}"
            save_dat_file(f"figures/FedAvg/fedavg_quant_{name}.dat", error)
        #ax.legend()
        ax.set(**pparam)
        fig.savefig("figures/FedAvg/convergence_fedavg_quantized.pdf", dpi=300)
        plt.close(fig)

        # FedPLT
        fig, ax = plt.subplots()
        for Delta in Delta_list:
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho, Delta=Delta)
            error = [la.norm(f.gradient(x_plt[..., k])) for k in range(x_plt.shape[-1])]
            label = "No quant." if Delta is None else fr"$\Delta={Delta}$"
            ax.semilogy(error, label=label)
            name = "noquant" if Delta is None else f"delta_{Delta}"
            save_dat_file(f"figures/FedPLT/fedplt_quant_{name}.dat", error)
        #ax.legend()
        ax.set(**pparam)
        fig.savefig("figures/FedPLT/convergence_fedplt_quantized.pdf", dpi=300)
        plt.close(fig)

        # Comparison
        for Delta in [1e-2, 1e-1]:
            fig, ax = plt.subplots()
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N, Delta=Delta)
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho, Delta=Delta)
            err_avg = [la.norm(f.gradient(x_avg[..., k])) for k in range(x_avg.shape[-1])]
            err_plt = [la.norm(f.gradient(x_plt[..., k])) for k in range(x_plt.shape[-1])]
            ax.semilogy(err_avg, label="FedAvg")
            ax.semilogy(err_plt, label="Fed-PLT")
            save_dat_file(f"figures/Compare/compare_fedavg_delta_{Delta}.dat", err_avg)
            save_dat_file(f"figures/Compare/compare_fedplt_delta_{Delta}.dat", err_plt)
            ax.legend(title=fr"$\Delta = {Delta}$")
            ax.set(**pparam)
            ax.set_title("FedAvg vs Fed-PLT under Quantization")
            fig.savefig(f"figures/Compare/compare_fedavg_fedplt_delta_{Delta}.pdf", dpi=300)
            plt.close(fig)