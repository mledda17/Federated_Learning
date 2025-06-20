# Federated Learning: FedAvg vs Fed-PLT

**Final report project for the EECI Course “Multi-Agent Optimization and Learning: Resilient and Adaptive Solutions”**

Author: Marco Ledda  
Institution: University of Cagliari

---

## 📖 Project Overview

This repository contains a full experimental framework for comparing **FedAvg** and **Fed-PLT** algorithms in Federated Learning under different practical scenarios:

- Varying local epochs \( N_e \)
- Communication quantization
- Stochastic gradients (mini-batch sizes)
- Logistic regression on synthetic classification tasks
- Evaluation of both **optimality convergence** and **classification accuracy**

The codebase allows reproducible experiments and exports TikZ-compatible `.dat` files for direct integration into LaTeX papers.

---

## 📊 Algorithms Implemented

- **FedAvg** ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629))  
- **Fed-PLT** ([Bastianello et al., 2024](https://arxiv.org/abs/2401.03849))

Both algorithms are implemented following their original formulations, with configurable local training epochs, quantization levels, and batch sizes.

---


