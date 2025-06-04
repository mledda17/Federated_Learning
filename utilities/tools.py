#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code by: Nicola Bastianello (KTH, nicolba@kth.se)

import numpy as np
from numpy import linalg as la
import networkx as net
import matplotlib.pyplot as plt

from scipy.special import expit

ran = np.random.default_rng()


# %% GRAPH TOOLS

def random_graph(N, edge_prob=0.25):
    """
    Generates an Erdos-Renyi graph with `N` nodes, and
    where each of the possible edges is chosen with
    probability `edge_prob` (the larger `edge_prob`, the
    more connected the graph).

    The function outputs the adjacency matrix of the
    graph, and plots it.
    """

    # ------ generate the graph
    is_connected = False
    while not is_connected:
        G = net.erdos_renyi_graph(N, edge_prob)
        is_connected = net.is_connected(G)

    # ------ get adjacency matrix
    adj_mat = net.adjacency_matrix(G).toarray()

    # ------ plot the graph
    plt.figure()
    plt.title(f"Erdos-Renyi random graph, {N} agents, edge prob. {edge_prob}")
    net.draw(G)
    plt.show()

    return adj_mat


def metropolis_hastings(adj_mat):
    N = adj_mat.shape[0]  # num. agents
    degrees = np.sum(adj_mat, axis=0)

    W = np.zeros((N, N))  # weights to be generated

    for i in range(N):
        # weigths for each neighbor
        for j in range(N):
            if adj_mat[i, j]:  # select neighbors only
                W[i, j] = 1 / (1 + max(degrees[i], degrees[j]))

        # self-weight
        W[i, i] = 1 - sum(W[i, :])

    return W


# %% COST FUNCTION

class LogisticRegression():

    def __init__(self, A, b, loss_weight=1, reg_weight=0, batch_sz=None):
        """
        Defines a logistic regression cost. Each row in `A` and
        corresponding element in `b` represent a data point.
        `loss_weight` and `reg_weight` multiply the loss and
        (l2) regularization respectively. `batch_sz` is the
        number of data points used to compute a gradient
        (if `None`, all data points are used).

        The attributes are: `shape` (size of model to be trained,
        deduced from `A`), `m` (num. of data points), `mod_smth`
        and `mod_scvx` the smoothness and strong convexity moduli.
        """

        # size of problem
        self.shape = (A.shape[1], 1)  # shape of the unknown model
        self.m = A.shape[0]  # num. data points

        # weight of loss and regularization terms
        self.loss_weight, self.reg_weight = loss_weight, reg_weight

        # batch size
        self.batch_sz = batch_sz

        # check if binary classification, and map classes to {-1,1}
        _c = np.unique(b)
        if len(_c) != 2: raise ValueError(f"Input data for a binary classification task, not for {len(_c)} classes.")
        b[np.where(b == _c[0])], b[np.where(b == _c[1])] = -1, 1

        # store data
        self.b = b.reshape((-1, 1))
        self.A = A

        # pre-compute useful matrix
        self._Ab = -np.multiply(self.A, self.b)

        # smoothness and strong convexity moduli
        self.mod_smth = self.loss_weight * self.m * np.max(
            [la.norm(self.A[[d],]) ** 2 for d in range(self.m)]) / 4 + self.reg_weight
        self.mod_scvx = 0 + self.reg_weight

    def gradient(self, x):

        # select which data points to use (use all if
        # `batch_size` is `None`)
        idx = range(self.m)
        if self.batch_sz is not None:
            idx = sorted(ran.choice(idx, size=int(self.batch_sz), replace=False))

        y = self._Ab[idx, :] * expit(self._Ab[idx, :].dot(x))

        return self.loss_weight * np.sum(y, axis=0, keepdims=True).T + self.reg_weight * x

    def proximal(self, x, penalty=1, tol=1e-5, max_iter=100):
        # approximate computation of prox using accelerated gradient descent

        l_max, l_min = self.mod_smth + 1 / penalty, self.mod_scvx + 1 / penalty
        c = (np.sqrt(l_max) - np.sqrt(l_min)) / (np.sqrt(l_max) + np.sqrt(l_min))

        w_old, y_old = x, x
        for _ in range(int(max_iter)):

            y = w_old - (self.gradient(w_old) + (w_old - x) / penalty) / l_max
            w = (1 + c) * y - c * y_old

            if la.norm(w - w_old) / (1e-20 + la.norm(w_old)) < tol: break

            w_old, y_old = w, y

        return w


# %% PLOT RESULTS

def plot_results_consensus(x):
    """
    Plot the evolution of the agents' states `x`, and
    their distance from the average consensus over time.
    `x` is assumed to have size (size state, num. agents, num. iterations).
    """

    # ------ extract info
    sz, N, num_iter = x.shape  # size state, num. agents, num. iterations
    x_avg = np.mean(x[..., 0])  # average consensus of initial conditions

    fig, ax = plt.subplots(1, 2, sharex=True)

    # ------ plot evolution of the states
    for i in range(N):
        ax[0].plot(x[:, i, :].flatten())

    # plot average consensus
    ax[0].plot([x_avg for _ in range(num_iter)], "--")

    ax[0].set_ylabel("Agents' states")

    # ------ plot distance from consensus
    ax[1].semilogy([la.norm(x[..., k] - x_avg) for k in range(num_iter)])

    ax[0].set_ylabel("Error")

    plt.show()


def plot_results_optimization(x, f):
    """
    Plot the evolution of the global gradient of `f` evaluated
    at the average of the local states.
    `x` is assumed to have size (size state, num. agents, num. iterations).
    """

    # ------ extract info
    N, num_iter = x.shape[-2], x.shape[-1]  # size state, num. agents, num. iterations

    # ------ plot results
    plt.figure()

    x_avg = [sum([x[..., i, k] for i in range(N)]) / N for k in range(num_iter)]

    plt.semilogy([la.norm(f.gradient(x_avg[k])) for k in range(num_iter)])

    plt.ylabel("Error")

    plt.show()


def plot_results_federated(x, f):
    """
    Plot the evolution of the global gradient of `f` evaluated
    at the state of the coordinator.
    `x` is assumed to have size (size state, num. iterations).
    """

    # ------ plot results
    plt.figure()

    plt.semilogy([la.norm(f.gradient(x[..., k])) for k in range(x.shape[-1])])

    plt.ylabel("Error")

    plt.show()



def accuracy(A, b, x):
    """
    Returns the ratio of correctly classified data points
    A: features, b: labels, x: model parameters
    """

    # classify all the data points in A
    cl = np.zeros(b.shape)

    for j in range(A.shape[0]):
        t = 1 / (1 + np.exp(-(A[j, :].dot(x))))
        cl[j] = 1 if t > 0.5 else -1  # store classified label

    # compare output of the model with actual labels, and count how many
    # are correctly classified
    diff = cl - b
    return 1 - np.count_nonzero(diff) / A.shape[0]