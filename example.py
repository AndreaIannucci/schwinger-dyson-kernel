import numpy as np

from src.sdker.solver import make_sdk_solver
from helper_functions import signature_increments_from_path

n_steps = 1_000
time = np.linspace(0.0, 1.0, n_steps + 1)

gamma = np.column_stack([
    np.sin(2.0 * np.pi * time),
    np.cos(2.0 * np.pi * time),
])

spec, solver = make_sdk_solver(
    path_dim=gamma.shape[1],
    depth=3,
)

increment_spec, increments = signature_increments_from_path(
    gamma,
    M=5,
    depth=3,
)

assert increment_spec == spec

G = solver.compute(increments)
print(G[0][-1][()])