# PINN-Gas-Lift: Physics-Informed NMPC for Niger Delta Optimization

[![PINNs vs Baselines](results/efficiency_bar.png)](results/efficiency_bar.png)
**20.1% lift efficiency gain • 13.43% gas savings • 95% accuracy • Real-time 0.261s solves**

Physics-Informed Neural Networks (PINNs) + Model Predictive Control (MPC) for gas-lift wells. Tackles 15-25% production losses in Niger Delta fields via physics-constrained optimization.

## 🎯 Key Results
| Metric | PINN-MPC | Conventional | Improvement |
|--------|----------|--------------|-------------|
| Lift Efficiency (STB/Mscf) | 20.1% ↑ | Baseline | **+20.1%** |
| Gas Consumption | ↓13.43% | Equal allocation | **$3.5M/year savings** (8-well) |
| Solve Time | 0.261s | Simulators | **10-20x faster** |
| Physics Violations | <0.62% | N/A | Constraint-safe |
| Oil Production | ±0.45% | Baseline | Maintained |

**MAPE:** BHP 0.18%, Water 3.55%, WHP 3.94%, Oil 5.79% [results/validation_plots/]

## 📄 Abstract 
> Gas-lift systems in Niger-Delta oil fields experience significant production inefficiencies, with estimated losses of 15-25% of potential oil production due to suboptimal gas allocation and inadequate control strategies. This paper presents a novel Physics-Informed Neural Network-Based Model Predictive Control (PINN-MPC) framework that integrates fundamental multiphase flow physics directly into a real-time optimization architecture...

## 🚀 Quick Start
```bash
pip install -r requirements.txt  # PyTorch, NumPy, SciPy
python src/train_pinn.py         # Train PINN model
python src/mpc_optimize.py       # Real-time NMPC


PINN-gas-lift/
├── src/              # train_pinn.py, mpc_optimize.py
├── data/             # Well sim data (CSVs)
├── results/          # Plots, validation metrics
├── notebooks/        # Analysis (optional)
└── requirements.txt

\frac{\partial p}{\partial t} + \nabla \cdot (\rho v) = 0

