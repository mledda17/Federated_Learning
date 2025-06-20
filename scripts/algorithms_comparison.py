import scienceplots
from utilities.save_dat_file import save_dat_file
from algorithms.fedavg import fed_avg
from algorithms.fedplt import fed_plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg as la



def comparison_varying_ne(f_i, f, v, step, num_iter, N, rho, A, b):
    N_e_list = [1, 10, 50, 100]
    pparam = dict(xlabel="Iterations", ylabel="Error")

    accuracy_fedavg_list, accuracy_fedplt_list = [], []

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

        fig, ax = plt.subplots()
        for N_e in N_e_list:
            x_avg = fed_avg(f_i, v, step, num_iter, N_e, N)
            xK = x_avg[..., -1].reshape((-1, 1))  # final model, shape (5, 1)

            # Predict labels
            probs = 1 / (1 + np.exp(-A @ xK))  # shape (1000, 1)
            preds = np.where(probs > 0.5, 1, -1)  # shape (1000, 1)
            accuracy_fedavg = np.mean(preds == b)

            accuracy_fedavg_list.append(accuracy_fedavg)
            ax.plot(N_e, accuracy_fedavg, "o", label=fr"$N_e = {N_e}$")
            save_dat_file(f"figures/FedAvg/fedavg_accuracy_ne_{N_e}.dat", [accuracy_fedavg])

        ax.set(xlabel=r"$N_e$", ylabel="Accuracy")
        ax.legend()
        fig.savefig("figures/FedAvg/accuracy_fedavg_varying_ne.pdf", dpi=300)
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

        fig, ax = plt.subplots()
        for N_e in N_e_list:
            x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho)
            xK = x_plt[..., -1].reshape((-1, 1))  # final model, shape (5, 1)

            # Predict labels
            probs = 1 / (1 + np.exp(-A @ xK))  # shape (1000, 1)
            preds = np.where(probs > 0.5, 1, -1)  # shape (1000, 1)
            accuracy_fedplt = np.mean(preds == b)

            accuracy_fedplt_list.append(accuracy_fedplt)
            ax.plot(N_e, accuracy_fedplt, "o", label=fr"$N_e = {N_e}$")
            save_dat_file(f"figures/FedPLT/fedplt_accuracy_ne_{N_e}.dat", [accuracy_fedplt])

        ax.set(xlabel=r"$N_e$", ylabel="Accuracy")
        ax.legend()
        fig.savefig("figures/FedPLT/accuracy_fedplt_varying_ne.pdf", dpi=300)
        plt.close(fig)

