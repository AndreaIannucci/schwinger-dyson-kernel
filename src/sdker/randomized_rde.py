from typing import Optional
import numpy as np
from scipy.linalg import expm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy.typing as npt


from typing import Dict, List, Tuple, Set, Optional


Array = npt.NDArray[np.floating]

def random_u_lie_algebra(N, rng):
    X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (X + X.conj().T) / 2
    A = 1j * H
    return A / np.sqrt(N)

def _one_rde_simulation(seed: int, path_increments: Array, N: int) -> complex:
    rng = np.random.default_rng(seed)
    d = path_increments.shape[0]
    A = [random_u_lie_algebra(N, rng) for _ in range(d)]
    M = sum(path_increments[i] * A[i] for i in range(d))
    U = expm(M)
    return np.trace(U)

@dataclass(frozen=True)
class RandomizedRDE:
    path_increments: Array  # shape (d,)
    matrix_dim: int         # N
    N_simul: int
    rng: Optional[np.random.Generator] = None

    def compute(self) -> complex:
        rng = self.rng
        if rng is None:
            rng = np.random.default_rng(0)

        N = self.matrix_dim

        unnormalized_out = 0.0 + 0.0j
        for _ in range(self.N_simul):
            unnormalized_out += _one_rde_simulation(rng, self.path_increments, N)        

        return unnormalized_out / (self.N_simul * N)


    def compute_parallel(self, n_workers=None) -> complex:
        rng = self.rng or np.random.default_rng(0)
        seeds = rng.integers(0, 2**32 - 1, size=self.N_simul)
        N = self.matrix_dim

        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_one_rde_simulation, seed, self.path_increments, N)
                for seed in seeds
            ]
            for future in as_completed(futures):
                results.append(future.result())

        return sum(results) / (self.N_simul * N)
