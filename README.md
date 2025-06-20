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

## 🔧 Project Structure

├── algorithms/
│ ├── fedavg.py # FedAvg algorithm
│ ├── fedplt.py # Fed-PLT algorithm
│
├── utilities/
│ ├── tools.py # Logistic regression model, gradient computation, accuracy
│ └── save_dat_file.py # Utility for exporting TikZ-compatible .dat files
│
├── run_scripts/
│ ├── algorithms_comparison.py # Comparison varying Ne
│ ├── run_quantization.py # Experiments with quantization
│ ├── run_stochastic_gradient.py # Experiments with stochastic gradients
│
├── figures/ # Output directory for plots and .dat files
│
└── main.py # Master script to run full experiments

---

## 🚀 Running the Experiments

### 1️⃣ Install required packages:

```bash
pip install -r requirements.txt
```

The only major dependencies are:
- numpy
- matplotlib
- scikit-learn
- scienceplots

### 2️⃣ Run full simulation pipeline:
Simply execute:
```bash
python main.py
```

This will:
* Generate synthetic classification data
* Run all algorithm under different conditions
* Export both .pdf plots and .dat files

  ## 📄 Report
This project has been used to produce the full simulation results and plots for the final academic report:

