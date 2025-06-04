import numpy as np
from utilities import tools
import os
from sklearn.datasets import make_classification
from scripts.run_quantization import run_quantization
from scripts.run_stochastic_gradient import run_stochastic_gradient
from scripts.algorithms_comparison import comparison_varying_ne

os.makedirs("figures/FedAvg", exist_ok=True)
os.makedirs("figures/FedPLT", exist_ok=True)
os.makedirs("figures/Compare", exist_ok=True)

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

f = tools.LogisticRegression(A, b, loss_weight=1/num_data, reg_weight=reg_weight, batch_sz=None)
f_i = [tools.LogisticRegression(A_i[i], b_i[i], loss_weight=1/num_data, reg_weight=reg_weight/N, batch_sz=None) for i in range(N)]

comparison_varying_ne(f_i, f, v, step, num_iter, N, rho)
run_quantization(f_i, f, v, step, num_iter, N, rho)
run_stochastic_gradient(A, b, A_i, b_i, v, step, num_iter, N, rho, reg_weight)

