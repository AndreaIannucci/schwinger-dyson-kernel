from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from typing import Dict, List, Tuple, Set, Optional

Array = npt.NDArray[np.floating]


@dataclass(frozen=True)
class SimpleSolver:
    """
    Explicit rectangle scheme for the iterated SD-equation.

    Approximates K(s,t) defined by:
        K(s,t) = 1 - sum_{i=1}^d int_{v=s}^t ( int_{u=s}^v K(s,u)K(u,v) dgamma_u^i ) dgamma_v^i
    """

    gamma: Array    # shape (N+1, d)

    def __post_init__(self) -> None:
        gamma = np.asarray(self.gamma, dtype=float)

        if gamma.ndim != 2:
            raise ValueError("gamma must have shape (N+1, d)")

        # freeze normalized array
        object.__setattr__(self, "gamma", gamma)

    def compute(self) -> Array:
        """
        Returns
        -------
        K : ndarray, shape (N+1, N+1)
            Upper-triangular matrix with K[a,a] = 1.
        """
       
        gamma = self.gamma
        n = gamma.shape[0] - 1
        dgamma = np.diff(gamma, axis=0)

        K = np.zeros((n + 1, n + 1), dtype=float)
        np.fill_diagonal(K, 1.0)

        for b in range(1, n + 1):
            # For a = b - 1, the inner sum is empty.
            K[b - 1, b] = 1.0

            if b == 1:
                continue

            # v_k = K[k,b-1] <Δgamma_k, Δgamma_{b-1}>
            v = (
                K[: b - 1, b - 1]
                * (dgamma[: b - 1] @ dgamma[b - 1])
            )

            # Compute all K[a,b], a = 0,...,b-2, simultaneously.
            K[: b - 1, b] = (
                K[: b - 1, b - 1]
                - K[: b - 1, : b - 1] @ v
            )

        return K