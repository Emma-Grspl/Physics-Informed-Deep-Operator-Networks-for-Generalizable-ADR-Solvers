# Physics-Informed DeepONet for the ADR Equation

This repository provides a high-performance implementation of a Physics-Informed Deep Operator Network (PI-DeepONet) designed to solve the 1D Advection-Diffusion-Reaction (ADR) equation. Unlike standard PINNs, this model learns the operator mapping from initial condition parameters to the full spatio-temporal solution, enabling real-time inference across a wide range of physical regimes.

## Physics Background

The model solves the non-linear ADR equation:

$$u_t + v u_x = D u_{xx} + \mu u (1 - u)$$

Where the dynamics are governed by:

* Advection ($v$): Pure transport of the signal.
* Diffusion ($D$):: Spatial spreading/dissipation.
* Reaction ($\mu$): Logistic growth representing local interactions.

## Model Architecture & Features

The architecture is designed to handle steep gradients and long-term temporal dependencies:

* Dual-Network Topology: A Branch Net (input: IC parameters, dim 9) and a Trunk Net (input: $(x, t)$ coordinates) fused via dot product.
* Fourier Feature Mapping: The Trunk Net uses periodic embeddings to mitigate the spectral bias of neural networks, allowing for the capture of high-frequency oscillations (e.g., Sin-Gauss profiles).
* Activation: SiLU (Swish) for smooth second-order derivatives in the PDE residual.
* Capacity: 256 latent dimensions with deep residual-like connections (5 layers for Branch, 4 for Trunk).

## Advanced Training Strategy

To reach a target error $< 3\%$ over a long time horizon ($t_{max} = 3.0$), we implemented several state-of-the-art strategies:

1. Temporal Curriculum (Time Marching): The model learns through successive time windows to prevent "Catastrophic Forgetting."
2. Dynamic Soft Polishing: A final "Elite" training phase using Active Learning. The batch composition is updated every 1000 iterations to focus on failing initial conditions (e.g., Sin-Gauss).
3. Hybrid Optimization: Seamless transition from Adam** (stochastic) to L-BFGS (second-order) to reach narrow local minima.
4. NTK Gradient Balancing: Adaptive weighting of loss components using the Neural Tangent Kernel (NTK) to resolve gradient pathologies between the PDE residual and Boundary Conditions.
5. King of the Hill: A robust rollback mechanism that monitors global L2 error and restores the best performing state, preventing divergence.

## Performance Analysis

The model is benchmarked against a Crank-Nicolson (CN) finite difference solver. The Mean L2 Relative Error is calculated over 1000 random physical configurations for each type of IC.

| Metric | Tanh | Sin-Gauss | Gaussian |
| --- | --- | --- | --- |
| **Mean L2 Error** | ~0.3% | ~2.9% | ~1.4% |

## Computational Efficiency: The Time-Jumping Advantage
Unlike classical finite difference solvers that must sequentially compute all intermediate time steps to reach a future state, the PI-DeepONet provides a continuous analytical solution. This allows the model to jump directly to the target time (t = 3.0) and process multiple scenarios in parallel on the GPU.

When evaluating the time required to predict the final state for 50 different physical scenarios:

- Crank-Nicolson (Sequential time-stepping): 0.726 s

- PI-DeepONet (Batched direct time-jumping): 0.034 s

Speedup: 21x Faster

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch (MPS/CUDA supported)
* Matplotlib, NumPy, PyYAML, tqdm

### Installation

```bash
git clone https://github.com/Emma-Grspl/These_DeepONet_ADR.git
cd These_DeepONet_ADR
pip install -r requirements.txt

```

### Running the Analysis

To evaluate the pre-trained model and generate the benchmark plots:

```bash
python src/analyse/global_analyse_PI_DeepOnet_vs_CN_vs_analytical.py

```
To run a complete training:

```bash
python scripts/train.py

```
## Repository Structure

* `assets/`: Main results.
* `configs/`: YAML files for hyperparameter management.
* `launch/`: Jean zay tickets (training was made on Jean-Zay, CNRS supercalculator)
* `models_saved/`: The trained and saved models.
* `outputs/`: Graphs and comparison between CN and PI-DeepOnet.
* `scripts/`: Files for launching classic training sessions or Optuna optimizations.
* `src/analyse/`: It features analysis between the PI-DeepOnet and the Crank Nicolson, as well as a plot function.
* `src/data/`: Generators for initial condition values and for batch training of the PI-DeepOnet.
* `src/models/`: Core PI-DeepONet implementation.
* `src/training/`: Core optimization architecture.
* `src/physics/`: PDE residual definitions and Autograd logic.
* `src/utils/`: Classical solvers (Crank-Nicolson) and metrics.
* `test/`: Allows verification of the validity of the calculation of pde_residual_ADR and of the validity of CN

---

**Author:** Emma Grospellier

**Project:** PhD Research 

---