# SDKer

[![Tests](https://github.com/AndreaIannucci/schwinger-dyson-kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreaIannucci/schwinger-dyson-kernel/actions/workflows/tests.yml)

Numerical research code for the Schwinger–Dyson kernel equation using path signatures, tensor-algebra expansions, and independent deterministic and randomized-RDE validation.

SDKer accompanies the paper [*Signature Kernel and Schwinger-Dyson Kernel Equations as Two-Parameter Rough Differential Equations*](https://arxiv.org/abs/2605.08844) by Thomas Cass, Dan Crisan, Andrea Iannucci, and William F. Turner.

> **Status:** SDKer is research software under active development. The current API should be regarded as experimental.

## Overview

The repository contains:

- a truncated tensor-algebra representation with graded word indexing;
- combinatorial generation and caching of inverse shuffles and non-crossing pairings;
- compilation of the Schwinger–Dyson polynomial into flat NumPy index arrays;
- a dynamic-programming solver with cached LU factorizations and Numba-accelerated polynomial evaluation;
- `SimpleSolver`, a direct first level discretization used as a reference method, first derived in the paper [*Free probability, path developments and signature kernels as universal scaling limits*](https://projecteuclid.org/journals/annals-of-applied-probability/volume-36/issue-2/Free-probability-path-developments-and-signature-kernels-as-universal-scaling/10.1214/25-AAP2240.short) by Thomas Cass and William F. Turner;

- `RandomizedRDE`, a Monte Carlo implementation used as an independent stochastic oracle;
- utilities for converting paths into block-signature increments; and
- an experiments/ directory contains path-generation utilities supporting the numerical examples considered in the paper;.

The expensive combinatorial polynomial is compiled once for a fixed path dimension and truncation depth. The compiled solver can then be reused across paths with the same configuration.

## Installation

SDKer requires Python 3.10. From the repository root, create and activate a virtual environment.

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The main dependencies are NumPy, SciPy, Numba, and `esig`.

## Quick start

The main solver consumes truncated signatures of path blocks rather than the raw path directly.

```python
import numpy as np

from sdker.signatures import signature_increments_from_path
from sdker.solver import make_sdk_solver


# A one-dimensional linear path.
gamma = np.linspace(0.0, 0.2, 21).reshape(-1, 1)

depth = 3
block_size = 1

# Compile once for this path dimension and truncation depth.
solution_spec, solver = make_sdk_solver(
    path_dim=gamma.shape[1],
    depth=depth,
)

# Convert gamma into truncated block signatures.
increment_spec, increments = signature_increments_from_path(
    path=gamma,
    M=block_size,
    depth=depth,
)

assert increment_spec == solution_spec

# Solve the interval recursion.
solution = solver.compute(increments)

# The scalar kernel is the empty-word coordinate on [0, T].
kernel_value = solution[0][-1][()]

print(kernel_value)
```

Polynomial compilation performs the expensive combinatorial preprocessing. For several paths with the same dimension and truncation depth, construct the solver once and reuse it; only the block signatures and solver.compute(...) call need to be repeated.

## Reference methods

### Deterministic reference solver

`SimpleSolver` operates directly on a discretized path:

```python
from sdker.reference_solver import SimpleSolver

K = SimpleSolver(gamma).compute()
reference_value = K[0, -1]
```

The SDK and reference solvers use different discretizations. They should not be expected to agree exactly on a coarse grid; the test suite verifies their agreement under mesh refinement. The SDK has the advantage of being able to handle paths of arbitrary low regularity whilst the reference solvers only cover the bounded variation regime. 

### Randomized-RDE oracle

For a straight-line path, the randomized-RDE estimator can be evaluated from its displacement:

```python
from sdker.randomized_rde import RandomizedRDE

displacement = gamma[-1] - gamma[0]

estimate = RandomizedRDE(
    path_increments=displacement,
    matrix_dim=4,
    N_simul=2_000,
    rng=np.random.default_rng(123),
).compute()
```

The estimator is complex-valued at finite Monte Carlo sample size. Its theoretical target is real, so a small nonzero imaginary component represents sampling error.

## Tests

Run the complete suite from the repository root:

```bash
python -m pytest
```

The suite covers:

- tensor-algebra indexing and products;
- inverse-shuffle and non-crossing-pairing combinatorics;
- cached index representations;
- compiled polynomial evaluation;
- block-signature construction and Chen's identity;
- deterministic and randomized reference methods;
- solver invariants and direct linear-solve checks;
- end-to-end agreement among the three numerical approaches; and
- covariance and scaling properties of the fractional Brownian motion simulator used in the experiments.

## Repository layout

```text
SDKer/
├── src/
│   └── sdker/
│       ├── combinatorics.py
│       ├── inverse_shuffle.py
│       ├── polynomial.py
│       ├── randomized_rde.py
│       ├── reference_solver.py
│       ├── shuffle.py
│       ├── signatures.py
│       ├── solver.py
│       └── tensor_algebra.py
├── tests/
├── examples/
├── pyproject.toml
├── LICENSE
└── README.md
```

Code under `src/sdker/` forms the reusable package. Utilities for generating fractional Brownian motion paths used to demonstrate and validate the solver in the numerical setting considered in the paper.

## Numerical considerations

- Computational and memory costs grow rapidly with the path dimension and tensor truncation depth.
- Polynomial compilation can be expensive, but its result is reusable for a fixed dimension and depth.
- The runtime solver stores a triangular table of tensor-valued interval solutions.
- `SimpleSolver` is intended primarily as an independent deterministic reference.
- `RandomizedRDE` is a Monte Carlo oracle and therefore requires statistical tolerances.
- The package is intended for research and reproducibility; it is not currently designed as production software.

## Citation

If this software contributes to academic work, please cite the accompanying paper:

```bibtex
@misc{cass2026signature,
  title         = {Signature Kernel and Schwinger-Dyson Kernel Equations as Two-Parameter Rough Differential Equations},
  author        = {Thomas Cass and Dan Crisan and Andrea Iannucci and William F. Turner},
  year          = {2026},
  eprint        = {2605.08844},
  archivePrefix = {arXiv},
  primaryClass  = {math.PR},
  doi           = {10.48550/arXiv.2605.08844}
}
```

## License

SDKer is released under the MIT License. See [`LICENSE`](LICENSE) for details.

