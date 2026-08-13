from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional

from .combinatorics import NonCrossingPairingCache, get_pairing_cache
from .tensor_algebra import TensorAlgebraSpec, TensorElement
from .inverse_shuffle import get_inverse_shuffle_cache, split_word

import numpy as np
from numba import njit


@njit(cache=True)
def _eval2_weighted_add_into_kernel(out, out_idx, g1_idx, g2_idx, gd1, gd2, w_inc):
    """
    out[out_idx[k]] += w_inc[k] * gd1[g1_idx[k]] * gd2[g2_idx[k]]
    """
    for k in range(out_idx.shape[0]):
        out[out_idx[k]] += w_inc[k] * gd1[g1_idx[k]] * gd2[g2_idx[k]]




@lru_cache(None)
def compiled_compute_vals_terms(
    coord: Tuple[int, ...],
    d_locked: int,
    gub_spec: "TensorAlgebraSpec",
    pairing_cache: "NonCrossingPairingCache",
    backward: bool = False
) -> Tuple[Tuple[int, float], ...]:
    """
    Parameters
    ----------
    
    Precompile:
        compute_vals(gub, coord, d_locked)
    as
        sum_j coeff_j * gub[idx_j]

    with coeff_j = (-1)^{#pairs}.

    coord: Tuple[int, ...]
           Word identifying the coordinate,
    d_locked: int
           Number of dimension locked,
    pairing_cache: "NonCrossingPairingCache"
           Lookup table for the coordinate,
    backward: bool
           True if the path is going backward in time when expanding.
    """
    n = len(coord)
    terms: List[Tuple[int, float]] = []

    for pairs, unmatched in pairing_cache.table_dn[n][d_locked]:
        # Kronecker constraints
        ok = True
        for (i, j) in pairs:
            if coord[i - 1] != coord[j - 1]:
                ok = False
                break
        if not ok:
            continue

        # Sign (-1)^{#pairs}
        sign = -1.0 if (len(pairs) % 2 == 1) else 1.0

        # sign for backward iteration
        if backward:
            sign *= -1.0 if (len(unmatched) % 2 == 1) else 1.0

        # Unmatched positions -> word
        unmatched_word = tuple(coord[k - 1] for k in unmatched)
        idx = gub_spec.word_to_index(unmatched_word)

        terms.append((idx, sign))

    return tuple(terms)


# ============================================================
# Compiled polynomial
# ============================================================

@dataclass(frozen=True)
class CompiledPolyNP:
    spec_out: "TensorAlgebraSpec"

    # terms: coeff * gub1[g1_idx] * gub2[g2_idx] * path[p_idx]
    out_idx: np.ndarray
    g1_idx: np.ndarray
    g2_idx: np.ndarray
    p_idx: np.ndarray
    coeff: np.ndarray

    def precompute_inc_weights(self, path: "TensorElement") -> np.ndarray:
        """
        w_inc = coeff * path[p_idx]
        Cache this per increment at runtime.
        """
        pd = path._data
        if self.coeff.size == 0:
            return np.empty((0,), dtype=np.float64)

        # Use float64 for a stable numba signature and predictable speed
        return (self.coeff.astype(np.float64, copy=False) * pd[self.p_idx].astype(np.float64, copy=False))

    # ---- NEW: fast path working on raw arrays (no TensorElement, no np.add.at) ----
    def eval2_weighted_add_into_data(
        self,
        out: np.ndarray,
        gd1: np.ndarray,
        gd2: np.ndarray,
        w_inc: np.ndarray,
    ) -> None:
        """
        Add poly(gd1, gd2, inc) into 'out' using precomputed w_inc.
        Uses a numba kernel for speed and no intermediate allocations.
        """
        if w_inc.size == 0:
            return
        _eval2_weighted_add_into_kernel(out, self.out_idx, self.g1_idx, self.g2_idx, gd1, gd2, w_inc)

    # ---- Backwards-compatible API ----
    def eval2(
        self,
        gub1: "TensorElement",
        gub2: "TensorElement",
        path: "TensorElement",
    ) -> "TensorElement":
        if gub1.spec != self.spec_out or gub2.spec != self.spec_out:
            raise ValueError("gub.spec mismatch")

        gd1 = gub1._data.astype(np.float64, copy=False)
        gd2 = gub2._data.astype(np.float64, copy=False)
        w_inc = self.precompute_inc_weights(path)

        out = np.zeros(self.spec_out.total_dim, dtype=np.float64)
        self.eval2_weighted_add_into_data(out, gd1, gd2, w_inc)
        return TensorElement(self.spec_out, out)

    def eval(self, gub: "TensorElement", path: "TensorElement") -> "TensorElement":
        return self.eval2(gub, gub, path)



# ============================================================
# Polynomial compiler 
# ============================================================

def compile_poly_numpy(
    gub_spec: "TensorAlgebraSpec",
    path_spec: "TensorAlgebraSpec",
    N: int,
    max_d: int,
) -> CompiledPolyNP:
    pairing_cache = get_pairing_cache(N, max_d)
    inv_cache = get_inverse_shuffle_cache(gub_spec, path_spec)

    gub_words = [gub_spec.index_to_word(i) for i in range(gub_spec.total_dim)]
    path_words = [path_spec.index_to_word(i) for i in range(path_spec.total_dim)]

    out_idx: List[int] = []
    g1_idx: List[int] = []
    g2_idx: List[int] = []
    p_idx: List[int] = []
    coeff: List[float] = []

    # ------------------------------------------------------------
    # Non-empty words only
    # ------------------------------------------------------------
    for out_i, w in enumerate(gub_words):
        if len(w) == 0:
            continue  # ignore empty output word completely

        pre, post, mid = split_word(w)

        for p_i, u in enumerate(path_words):
            mu = (mid,) + u
            if len(mu) > gub_spec.max_level:
                continue

            mu_idx = gub_spec.word_to_index(mu)

            for i1, i2 in inv_cache.table[out_i][mu_idx]:
                left_word  = gub_words[i1]
                right_word = gub_words[i2]

                left_terms = compiled_compute_vals_terms(
                    left_word, len(pre), gub_spec, pairing_cache
                )
                right_terms = compiled_compute_vals_terms(
                    right_word, len(post), gub_spec, pairing_cache, backward=True
                )

                for l_idx, lc in left_terms:
                    for r_idx, rc in right_terms:
                        out_idx.append(out_i)
                        g1_idx.append(l_idx)
                        g2_idx.append(r_idx)
                        p_idx.append(mu_idx)
                        coeff.append(-lc * rc)

    return CompiledPolyNP(
        spec_out=gub_spec,
        out_idx=np.asarray(out_idx, dtype=np.int64),
        g1_idx=np.asarray(g1_idx, dtype=np.int64),
        g2_idx=np.asarray(g2_idx, dtype=np.int64),
        p_idx=np.asarray(p_idx, dtype=np.int64),
        coeff=np.asarray(coeff, dtype=np.float64),
    )
