import scienceplots
from utilities import tools
from utilities.save_dat_file import save_dat_file
from algorithms.fedavg import fed_avg
from algorithms.fedplt import fed_plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as la

def run_stochastic_gradient_fedavg(A, b, A_i, b_i, v, step, num_iter, N, reg_weight, num_data, Ne_list):
    batch_sizes = [None, int(num_data/2), int(num_data/4), 1]
    N_e = 1
    pparam = dict(xlabel="Iterations", ylabel="Error")

    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        for N_e in Ne_list:
            fig_avg, ax_avg = plt.subplots()

            # Store accuracy for each batch size
            accuracy_avg_list = []

            for batch_sz in batch_sizes:
                # Global cost function (only for gradient evaluation)
                f = tools.LogisticRegression(A, b, loss_weight=1 / 100, reg_weight=reg_weight)

                # Local models
                f_i = [tools.LogisticRegression(A_i[i], b_i[i], loss_weight=1 / 100,
                                                 reg_weight=reg_weight / N, batch_sz=batch_sz)
                       for i in range(N)]

                # Run FedAvg
                x_avg = fed_avg(f_i, v, step, num_iter, N_e, N)

                # Compute optimality
                err_avg = [la.norm(f.gradient(x_avg[..., k])) for k in range(num_iter + 1)]

                # Compute final accuracy
                acc = tools.accuracy(A, b, x_avg[..., -1])
                accuracy_avg_list.append(acc)
                print(f"FedAvg: Batch={batch_sz}, N_e={N_e}, Accuracy={acc:.4f}")

                # Plot optimality
                label = "Full batch" if batch_sz is None else f"Batch={batch_sz}"
                ax_avg.semilogy(err_avg, label=label)

                # Save optimality .dat
                name = "full" if batch_sz is None else f"batch_{int(batch_sz)}"
                save_dat_file(f"figures/FedAvg/fedavg_sgd_{name}_Ne_{N_e}.dat", err_avg)

            # Plot customization
            ax_avg.legend(title="Batch size")
            ax_avg.set(**pparam)
            ax_avg.set_title(f"FedAvg with Stochastic Gradients (Ne={N_e})")
            fig_avg.savefig(f"figures/FedAvg/convergence_fedavg_sgd_Ne_{N_e}.pdf", dpi=300)
            plt.close(fig_avg)

            # Save accuracy .dat file
            batch_labels = ["full" if b is None else str(int(b)) for b in batch_sizes]
            save_dat_file(f"figures/FedAvg/accuracy_fedavg_Ne_{N_e}.dat", accuracy_avg_list)


def run_stochastic_gradient_fedplt(A, b, A_i, b_i, v, step, num_iter, N, reg_weight_list, rho, num_data, Ne_list):
    batch_sizes = [None, int(num_data/2), int(num_data/4), 1]
    pparam = dict(xlabel="Iterations", ylabel="Error")

    with plt.style.context(["science", "ieee"]):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]

        for N_e in Ne_list:
            # We will store accuracy for each batch size and reg_weight
            accuracy_dict = {str(batch_sz): [] for batch_sz in batch_sizes}

            for reg_weight in reg_weight_list:
                fig, ax = plt.subplots()

                for batch_sz in batch_sizes:
                    f = tools.LogisticRegression(A, b, loss_weight=1 / 100, reg_weight=reg_weight)
                    f_i = [tools.LogisticRegression(A_i[i], b_i[i], loss_weight=1 / 100,
                                                     reg_weight=reg_weight / N, batch_sz=batch_sz)
                           for i in range(N)]

                    x_plt = fed_plt(f_i, v, step, num_iter, N_e, N, rho)
                    err_plt = [la.norm(f.gradient(x_plt[..., k])) for k in range(num_iter + 1)]

                    # Save optimality dat file
                    name = "full" if batch_sz is None else f"batch_{batch_sz}"
                    save_dat_file(f"figures/FedPLT/fedplt_sgd_{name}_Ne_{N_e}_reg_{reg_weight}.dat", err_plt)

                    # Plot optimality
                    label = "Full batch" if batch_sz is None else f"Batch={batch_sz}"
                    ax.semilogy(err_plt, label=label)

                    # Compute accuracy at final model
                    acc = tools.accuracy(A, b, x_plt[..., -1])
                    accuracy_dict[str(batch_sz)].append(acc)
                    print(f"Accuracy for batch {batch_sz}, N_e={N_e}, reg_weight={reg_weight}: {acc:.4f}")

                ax.legend(title="Batch size")
                ax.set(**pparam)
                ax.set_title(f"Fed-PLT: $N_e={N_e}$, reg={reg_weight}")
                fig.savefig(f"figures/FedPLT/convergence_fedplt_sgd_Ne_{N_e}_reg_{reg_weight}.pdf", dpi=300)
                plt.close(fig)

            # After sweeping reg_weight, plot accuracy vs reg_weight
            fig_acc, ax_acc = plt.subplots()
            for batch_sz in batch_sizes:
                batch_label = "Full batch" if batch_sz is None else f"Batch={batch_sz}"
                accuracy_values = accuracy_dict[str(batch_sz)]
                ax_acc.plot(reg_weight_list, accuracy_values, marker='o', label=batch_label)
                # Save accuracy dat file
                save_dat_file(f"figures/FedPLT/accuracy_batch_{batch_sz}_Ne_{N_e}.dat", accuracy_values)

            ax_acc.set_xlabel("Regularization weight")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title(f"Fed-PLT Accuracy vs RegWeight (Ne={N_e})")
            ax_acc.legend()
            fig_acc.savefig(f"figures/FedPLT/accuracy_vs_regweight_Ne_{N_e}.pdf", dpi=300)
            plt.close(fig_acc)