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
- `qsvt-linear-solver-eig-plot.ipynb`: plot results for eigenvalue / singular value experiments
  - check whether small-degree polynomials can preserve eigenvalues / singular values

## Bug (Fixed)

Due to some numerical issues, arbitrary matrix may fail to be block-encoded for `linear_solver`. To experiment with `linear_solver`, it is recommended to use [circulant matrices](https://en.wikipedia.org/wiki/Circulant_matrix).

> **Fix (2023/08/29):**  [`pennylane.BlockEncode()`](https://docs.pennylane.ai/en/stable/code/api/pennylane.BlockEncode.html) is utilized instead for accurate block-encoding. Old codes need not be modified.

> **Fix again (2023/09/08)**: It turns out that there is a **bug** in my implementation of `block_encode()`, which is not a numerical one. After fixing it, `block_encode()` works well, and thus `pennylane.BlockEncode()` is not required anymore.