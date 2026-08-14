from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .combinatorics import get_pairing_cache
from .polynomial import (
    CompiledPolyNP,
    compile_poly_numpy,
    compiled_compute_vals_terms,
)
from .tensor_algebra import TensorAlgebraSpec, TensorElement


@dataclass(frozen=True)
class SDKSolverCompiled:
    """
    Compile-time data for the SDK solver.

    This object stores:
      - the compiled polynomial defining the recursion,
      - algebraic constants,
      - expensive algebraic reductions invariant across executions.

    Once constructed, this object is immutable and can be reused
    for many different paths.
    """

    poly: CompiledPolyNP
    gub_spec: TensorAlgebraSpec
    max_d: int

    def __post_init__(self):
        object.__setattr__(self, "D", self.gub_spec.total_dim)
        object.__setattr__(
            self,
            "empty_idx",
            self.gub_spec.word_to_index(()),
        )
        object.__setattr__(
            self,
            "Id",
            TensorElement.eye(self.gub_spec),
        )
        object.__setattr__(
            self,
            "N",
            self.gub_spec.max_level,
        )

        object.__setattr__(
            self,
            "_empty_rule",
            self._precompute_empty_rule(),
        )
        object.__setattr__(
            self,
            "_lin_part",
            self._right_linear_part(),
        )

    def _precompute_empty_rule(self):
        """
        Precompute the algebraic rule governing the empty-word component.

        This encodes:
            x[()] = previous_x[()]
                    + sum_k sign_k * inc[p_k] * x[col_k]
        """
        pairing_cache = get_pairing_cache(
            self.N,
            self.max_d,
        )

        empty_cols = []
        empty_pidx = []
        empty_sign = []

        for p_i in range(self.D):
            word = self.gub_spec.index_to_word(p_i)

            if len(word) == 0:
                continue

            terms = compiled_compute_vals_terms(
                word,
                d_locked=0,
                gub_spec=self.gub_spec,
                pairing_cache=pairing_cache,
                backward=False,
            )

            for col_idx, sign in terms:
                empty_cols.append(col_idx)
                empty_pidx.append(p_i)
                empty_sign.append(float(sign))

        empty_cols = np.asarray(
            empty_cols,
            dtype=np.int64,
        )
        empty_pidx = np.asarray(
            empty_pidx,
            dtype=np.int64,
        )
        empty_sign = np.asarray(
            empty_sign,
            dtype=np.float64,
        )

        empty_rows = np.full(
            empty_cols.shape[0],
            self.empty_idx,
            dtype=np.int64,
        )

        return (
            empty_rows,
            empty_cols,
            empty_pidx,
            empty_sign,
        )

    def _right_linear_part(self):
        """
        Extract the part of the polynomial linear in the unknown.

        The unknown x = G(i, j + 1) appears as the second tensor
        argument. Therefore, we retain terms whose first tensor
        coordinate is the empty word.
        """
        mask = self.poly.g1_idx == self.empty_idx

        return (
            self.poly.out_idx[mask],
            self.poly.g2_idx[mask],
            self.poly.p_idx[mask],
            self.poly.coeff[mask],
        )


class SDKSolverRuntime:
    """
    Runtime solver for a fixed compiled SDK problem.

    Given a sequence of path increments, this object computes the
    triangular table G[i][j] using interval-based dynamic programming
    and linear solves.
    """

    def __init__(self, compiled: SDKSolverCompiled):
        self.compiled = compiled

    def compute(
        self,
        path_increments: list[TensorElement],
    ):
        """
        Compute the triangular SDK solution table.

        The implementation uses:
          - one precomputed polynomial-weight array per increment,
          - one LU factorization per time index,
          - raw NumPy arrays internally.
        """
        D = self.compiled.D
        T = len(path_increments)

        empty_idx = self.compiled.empty_idx
        identity = self.compiled.Id
        spec = self.compiled.gub_spec

        for index, increment in enumerate(path_increments):
            if not isinstance(increment, TensorElement):
                raise TypeError(
                    f"path_increments[{index}] must be a TensorElement"
                )

            if increment.spec != spec:
                raise ValueError(
                    f"path_increments[{index}] has incompatible spec: "
                    f"expected {spec}, got {increment.spec}"
                )

        identity_data = identity._data.astype(
            np.float64,
            copy=False,
        )
        identity_matrix = np.eye(
            D,
            dtype=np.float64,
        )

        (
            empty_rows,
            empty_cols,
            empty_pidx,
            empty_sign,
        ) = self.compiled._empty_rule

        (
            out_lin,
            col_lin,
            p_lin,
            c_lin,
        ) = self.compiled._lin_part

        poly = self.compiled.poly

        # These values do not change during the dynamic program.
        linear_coefficients = c_lin.astype(
            np.float64,
            copy=False,
        )

        # Each increment is reused many times. Compute its polynomial
        # weights once and access them directly by time index.
        increment_weights = tuple(
            poly.precompute_inc_weights(increment)
            for increment in path_increments
        )

        def build_A(
            increment: TensorElement,
        ) -> np.ndarray:
            """
            Build the linear operator associated with one increment.
            """
            increment_data = increment._data.astype(
                np.float64,
                copy=False,
            )

            weights = (
                linear_coefficients
                * increment_data[p_lin]
            )

            A = np.zeros(
                (D, D),
                dtype=np.float64,
            )

            np.add.at(
                A,
                (out_lin, col_lin),
                weights,
            )

            return A

        def factorize_increment(
            increment: TensorElement,
        ):
            """
            Build and factorize M = I - A(increment), enforcing
            the empty-word row rule.
            """
            A = build_A(increment)
            M = identity_matrix - A

            increment_data = increment._data.astype(
                np.float64,
                copy=False,
            )

            empty_row_values = (
                empty_sign
                * increment_data[empty_pidx]
            )

            M[empty_idx, :] = 0.0
            M[empty_idx, empty_idx] = 1.0

            np.add.at(
                M,
                (empty_rows, empty_cols),
                -empty_row_values,
            )

            return lu_factor(M)

        # Factor every time-index matrix before entering the
        # dynamic-programming loops.
        lu_factors = tuple(
            factorize_increment(increment)
            for increment in path_increments
        )

        # Raw internal table. Diagonal entries can safely share the
        # identity array because they are never mutated.
        G_data = [
            [None for _ in range(T + 1)]
            for _ in range(T + 1)
        ]

        for k in range(T + 1):
            G_data[k][k] = identity_data

        # Compute intervals in increasing order of length.
        for length in range(1, T + 1):
            for i in range(T + 1 - length):
                j = i + length - 1

                lu, piv = lu_factors[j]

                rhs = np.zeros(
                    D,
                    dtype=np.float64,
                )

                # First contribution:
                # poly(G[i][j], identity, increment[j]).
                poly.eval2_weighted_add_into_data(
                    rhs,
                    G_data[i][j],
                    identity_data,
                    increment_weights[j],
                )

                # Remaining interval contributions.
                for m in range(i + 1, j + 1):
                    poly.eval2_weighted_add_into_data(
                        rhs,
                        G_data[i][m - 1],
                        G_data[m][j + 1],
                        increment_weights[m - 1],
                    )

                rhs[empty_idx] = G_data[i][j][empty_idx]

                G_data[i][j + 1] = lu_solve(
                    (lu, piv),
                    rhs,
                )

        # Convert raw arrays back to the existing public API.
        G = [
            [None for _ in range(T + 1)]
            for _ in range(T + 1)
        ]

        for i in range(T + 1):
            for j in range(T + 1):
                data = G_data[i][j]

                if data is not None:
                    G[i][j] = TensorElement(
                        spec,
                        data,
                    )

        return G


def make_sdk_solver(
    path_dim: int,
    depth: int,
) -> tuple[TensorAlgebraSpec, SDKSolverRuntime]:
    """
    Compile an SDK solver for a fixed path dimension and tensor depth.

    Compilation performs the expensive combinatorial preprocessing.
    Reuse the returned solver for paths sharing the same dimension
    and truncation depth.
    """
    if path_dim < 1:
        raise ValueError("path_dim must be positive")

    if depth < 1:
        raise ValueError("depth must be positive")

    solution_spec = TensorAlgebraSpec(
        dim=path_dim,
        max_level=depth,
    )

    tail_spec = TensorAlgebraSpec(
        dim=path_dim,
        max_level=depth - 1,
    )

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=depth,
        max_d=depth,
    )

    compiled = SDKSolverCompiled(
        poly=poly,
        gub_spec=solution_spec,
        max_d=depth,
    )

    return (
        solution_spec,
        SDKSolverRuntime(compiled),
    )