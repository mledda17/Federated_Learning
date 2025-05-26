import numpy as np
import tools
from sklearn.datasets import make_classification
from scripts.fedavg import fedavg
from scripts.fedplt import fedplt

ran = np.random.default_rng()

# Problem size
N = 10                   # num. agents
num_data = 100           # per agent
num_features = 5         # size of model to be trained
eps = 50
step = 0.01
N_e = 1
num_iter = 200
rho = 1.0

# randomly generate data

A, b = make_classification(n_samples=num_data*N, n_features=num_features, n_clusters_per_class=1)

b = b.reshape((-1,1))
b[np.where(b == 0)], b[np.where(b == 1)] = -1, 1

# divide data by agents
A_i = [A[i*num_data:(i+1)*num_data,:] for i in range(N)]
b_i = [b[i*num_data:(i+1)*num_data,:] for i in range(N)]

# create global cost and local costs
# NOTE: the global `reg_weight` is divided equally between agents (reg_weight/N)
f = tools.LogisticRegression(A, b, loss_weight=1/num_data, reg_weight=50)
f_i = [tools.LogisticRegression(A_i[i], b_i[i], loss_weight=1/num_data, reg_weight=50/N) for i in range(N)]

# generate the initial states of the agents
v = ran.normal(size=f.shape)

# Run FedAvg
x_hist_avg = fedavg(f_i, v, step, num_iter, N_e, N)

# Run Fed-PLT
x_hist_plt = fedplt(f_i, v, step, num_iter, N_e, N, rho)

# Plot Results
print("FedAvg Results:")
tools.plot_results_federated(x_hist_avg.T, f)

print("Fed-PLT Results:")
tools.plot_results_federated(x_hist_plt.T, f)