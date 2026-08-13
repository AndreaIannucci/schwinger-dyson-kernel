from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .polynomial import compiled_compute_vals_terms, CompiledPolyNP, compile_poly_numpy
from .combinatorics import get_pairing_cache
from .tensor_algebra import TensorAlgebraSpec, TensorElement


@dataclass(frozen=True)
class SDKSolverCompiled:
    """
    Compile-time data for the SDK solver.

    This object stores:
      - the compiled polynomial defining the recursion,
      - algebraic constants (dimension, empty word, identity),
      - all expensive algebraic reductions that are invariant
        across runtime executions.

    Once constructed, this object is immutable and can be reused
    for many different paths.
    """
    poly: CompiledPolyNP
    gub_spec: TensorAlgebraSpec
    # N: int  # dimension of solution tensor algebra
    max_d: int   # information about the maximum index popped 

    def __post_init__(self):
        # --------------------------------------------------------
        # Basic algebraic constants
        # --------------------------------------------------------
        object.__setattr__(self, "D", self.gub_spec.total_dim)
        object.__setattr__(self, "empty_idx", self.gub_spec.word_to_index(()))
        object.__setattr__(self, "Id", TensorElement.eye(self.gub_spec))
        object.__setattr__(self, "N", self.gub_spec.max_level)
        # --------------------------------------------------------
        # Expensive precomputations
        #
        # 1) empty-word update rule (row replacement)
        # 2) right-linear part of the polynomial (matrix A)
        # --------------------------------------------------------
        object.__setattr__(self, "_empty_rule", self._precompute_empty_rule())
        object.__setattr__(self, "_lin_part", self._right_linear_part())

    # ============================================================
    # Empty-word rule
    # ============================================================

    def _precompute_empty_rule(self):
        """
        Precompute the algebraic rule governing the empty-word component.

        This encodes the identity:
            x[()] = previous_x[()]
                    + sum_k sign_k * inc[p_k] * x[col_k]

        Runtime enforcement can then be done with a single np.add.at call
        without allocating any row-index array inside the DP loop.
        """
        pairing_cache = get_pairing_cache(self.N, self.max_d)

        empty_cols = []
        empty_pidx = []
        empty_sign = []

        for p_i in range(self.D):
            word = self.gub_spec.index_to_word(p_i)
            if len(word) == 0:
                continue

            # Linear expansion induced by non-crossing pairings
            terms = compiled_compute_vals_terms(
                word,
                d_locked=0,
                gub_spec=self.gub_spec,
                pairing_cache=pairing_cache,
                backward=False,
            )

            for col_idx, sgn in terms:
                empty_cols.append(col_idx)
                empty_pidx.append(p_i)
                empty_sign.append(float(sgn))

        empty_cols = np.asarray(empty_cols, dtype=np.int64)
        empty_pidx = np.asarray(empty_pidx, dtype=np.int64)
        empty_sign = np.asarray(empty_sign, dtype=np.float64)

        # Precompute the row indices once (used in np.add.at)
        empty_rows = np.full(empty_cols.shape[0], self.empty_idx, dtype=np.int64)

        return (empty_rows, empty_cols, empty_pidx, empty_sign)


    # ============================================================
    # Linear part extraction
    # ============================================================

    def _right_linear_part(self):
        """
        Extract the part of the polynomial that is linear in the unknown.

        In the recurrence, the unknown x = G(i, j+1) always appears
        as the second tensor argument of poly.eval2. Therefore,
        the linear operator A is formed by keeping only those terms
        where gub1 is the empty word. This is because G(i,i) = 0 for every 
        word of positive length.
        """
        mask = self.poly.g1_idx == self.empty_idx

        return (
            self.poly.out_idx[mask],
            self.poly.g2_idx[mask],
            self.poly.p_idx[mask],
            self.poly.coeff[mask],
        )


# ============================================================
# Runtime solver
# ============================================================

class SDKSolverRuntime:
    """
    Runtime solver for a fixed compiled SDK problem.

    Given a sequence of path increments, this object computes
    the triangular table G[i][j] via interval-based dynamic
    programming and linear solves.
    """

    def __init__(self, compiled: SDKSolverCompiled):
        self.compiled = compiled
    
    def compute(self, path_increments: list[TensorElement]):
        """
        DP solve with:
        - LU factorization cached per j (per increment)
        - Polynomial evaluation using cached w_inc = coeff * inc[p_idx]
        - Internal storage of G as raw numpy arrays (avoids TensorElement churn)
        """
        D = self.compiled.D
        T = len(path_increments)

        empty_idx = self.compiled.empty_idx
        Id = self.compiled.Id
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

        Id_data = Id._data.astype(np.float64, copy=False)
        Id_mat = np.eye(D, dtype=np.float64)

        # Unpack precomputed data
        empty_rows, empty_cols, empty_pidx, empty_sign = self.compiled._empty_rule
        out_lin, col_lin, p_lin, c_lin = self.compiled._lin_part
        poly = self.compiled.poly

        # --------------------------------------------------------
        # Allocate DP table (raw arrays) and initialize diagonal
        # --------------------------------------------------------
        G_data = [[None for _ in range(T + 1)] for __ in range(T + 1)]
        for k in range(T + 1):
            G_data[k][k] = Id_data  # safe to share; we never mutate these vectors

        # --------------------------------------------------------
        # Caches
        # --------------------------------------------------------
        lu_cache = {}   # id(inc_j) -> (lu, piv)
        w_cache = {}    # id(inc_m) -> w_inc = coeff * inc[p_idx]

        def get_w(inc: TensorElement) -> np.ndarray:
            key = id(inc)
            w = w_cache.get(key)
            if w is None:
                w = poly.precompute_inc_weights(inc)
                w_cache[key] = w
            return w

        def build_A(inc: TensorElement) -> np.ndarray:
            """
            Build the linear operator A associated with a path increment inc.
            """
            pd = inc._data
            weights = c_lin.astype(np.float64, copy=False) * pd[p_lin].astype(np.float64, copy=False)

            A = np.zeros((D, D), dtype=np.float64)
            np.add.at(A, (out_lin, col_lin), weights)
            return A

        def get_lu_for_increment(inc: TensorElement):
            """
            Build and factorize M = (I - A(inc)) with empty-row rule enforced.
            Cached per increment (per j).
            """
            key = id(inc)
            if key in lu_cache:
                return lu_cache[key]

            A = build_A(inc)
            M = Id_mat - A

            # Enforce: x_empty - sum(empty_row_vals * x_col) = b_empty
            pd = inc._data.astype(np.float64, copy=False)
            empty_row_vals = empty_sign * pd[empty_pidx]

            M[empty_idx, :] = 0.0
            M[empty_idx, empty_idx] = 1.0
            np.add.at(M, (empty_rows, empty_cols), -empty_row_vals)

            lu_cache[key] = lu_factor(M)
            return lu_cache[key]

        # ========================================================
        # Main interval DP
        # ========================================================
        for L in range(1, T + 1):
            for i in range(T + 1 - L):
                j = i + L - 1  # compute G[i][j+1]

                inc_self = path_increments[j]
                lu, piv = get_lu_for_increment(inc_self)

                # ------------------------------------------------
                # Build RHS b (no intermediate TensorElements)
                # ------------------------------------------------
                b = np.zeros(D, dtype=np.float64)

                # b += poly(G[i][j], Id, inc_self)
                poly.eval2_weighted_add_into_data(
                    b,
                    G_data[i][j],
                    Id_data,
                    get_w(inc_self),
                )

                # b += sum_m poly(G[i][m-1], G[m][j+1], inc_m)
                for m in range(i + 1, j + 1):
                    inc_m = path_increments[m - 1]
                    poly.eval2_weighted_add_into_data(
                        b,
                        G_data[i][m - 1],
                        G_data[m][j + 1],
                        get_w(inc_m),
                    )

                # empty-word RHS component
                b[empty_idx] = G_data[i][j][empty_idx]

                # ------------------------------------------------
                # Solve using cached LU
                # ------------------------------------------------
                x = lu_solve((lu, piv), b)
                G_data[i][j + 1] = x

        # --------------------------------------------------------
        # Wrap back into TensorElements (keep your original API)
        # --------------------------------------------------------
        G = [[None for _ in range(T + 1)] for __ in range(T + 1)]
        for a in range(T + 1):
            for b_ in range(T + 1):
                arr = G_data[a][b_]
                if arr is None:
                    G[a][b_] = None
                else:
                    G[a][b_] = TensorElement(spec, arr)

        return G




def make_sdk_solver(path_dim: int, depth: int) -> tuple[
    TensorAlgebraSpec,
    SDKSolverRuntime,
]:
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

    return solution_spec, SDKSolverRuntime(compiled)