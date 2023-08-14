# QSVT Implementation

This repository consists of implementations of the main framework of Quantum Singular Value Transformation (QSVT), and some algorithms based on it.

## Implemented Algorithms
- Amplitude amplification (search)
- Matrix inversion (for solving Quantum Linear System Problem, QLSP)
- Phase estimation (simplified)

## Notes on Repository Organization

- `experiments/`: contains some numerical experiment results of solving QLSP by QSVT (not organized yet)
- `qsvt/`: 
  - `Solvers/`: QSP angle solver, forked from [here](https://github.com/bartubisgin/QSVTinQiskit-2021-Europe-Hackathon-Winning-Project-)
  - `algorithms.py`: algorithms mentioned in [Implemented Algorithms](./README.md/#implemented-algorithms)
  - `core.py`: core functions, including QSVT, block-encoding, angle convention convertion, etc.
  - `helper.py`: helper functions for polynomial coefficient finding, calculating total variation distance, generating random matrices, etc.
  - `inv_k*_d*.txt`: polynomial coefficients (approximating $1/x$)
    - `k50`: condition number ($\kappa$) is $50$
    - `d601`: polynomial degree is $601$
- `plot.ipynb`: plotting experiment results
- `qsvt-aa.ipynb`: example for amplitude amplfication
- `qsvt-linear-solver.ipynb`: example for solving QLSP
- `qsvt-qpe.ipynb`: example for QPE
- `qsvt-linear-solver*.py`: more complex example for solving QLSP
- `*run.sh`: running experiements