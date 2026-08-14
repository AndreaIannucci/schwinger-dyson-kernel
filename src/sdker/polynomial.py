from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

from .combinatorics import (
    NonCrossingPairingCache,
    get_pairing_cache,
)
from .inverse_shuffle import (
    get_inverse_shuffle_cache,
    split_word,
)
from .tensor_algebra import (
    TensorAlgebraSpec,
    TensorElement,
)


@njit(cache=True)
def _eval2_weighted_add_into_kernel(
    out,
    out_idx,
    g1_idx,
    g2_idx,
    gd1,
    gd2,
    w_inc,
):
    """
    Evaluate the compiled polynomial and accumulate it into out.

    For every compiled term k:

        out[out_idx[k]] += (
            w_inc[k]
            * gd1[g1_idx[k]]
            * gd2[g2_idx[k]]
        )
    """
    for k in range(out_idx.shape[0]):
        out[out_idx[k]] += (
            w_inc[k]
            * gd1[g1_idx[k]]
            * gd2[g2_idx[k]]
        )


@lru_cache(None)
def compiled_compute_vals_terms(
    coord: Tuple[int, ...],
    d_locked: int,
    gub_spec: TensorAlgebraSpec,
    pairing_cache: NonCrossingPairingCache,
    backward: bool = False,
) -> Tuple[Tuple[int, float], ...]:
    """
    Precompile compute_vals(gub, coord, d_locked) as:

        sum_j coefficient_j * gub[index_j]

    where each coefficient is determined by the corresponding
    non-crossing pairing.
    """
    n = len(coord)
    terms: List[Tuple[int, float]] = []

    for pairs, unmatched in pairing_cache.table_dn[n][d_locked]:
        valid = True

        for i, j in pairs:
            if coord[i - 1] != coord[j - 1]:
                valid = False
                break

        if not valid:
            continue

        sign = (
            -1.0
            if len(pairs) % 2 == 1
            else 1.0
        )

        if backward and len(unmatched) % 2 == 1:
            sign *= -1.0

        unmatched_word = tuple(
            coord[k - 1]
            for k in unmatched
        )

        index = gub_spec.word_to_index(
            unmatched_word
        )

        terms.append(
            (index, sign)
        )

    return tuple(terms)


@dataclass(frozen=True)
class CompiledPolyNP:
    """
    Sparse compiled representation of the SDK polynomial.

    Each term has the form:

        coeff[k]
        * gub1[g1_idx[k]]
        * gub2[g2_idx[k]]
        * path[p_idx[k]]

    and contributes to out[out_idx[k]].
    """

    spec_out: TensorAlgebraSpec
    out_idx: np.ndarray
    g1_idx: np.ndarray
    g2_idx: np.ndarray
    p_idx: np.ndarray
    coeff: np.ndarray

    def precompute_inc_weights(
        self,
        path: TensorElement,
    ) -> np.ndarray:
        """
        Precompute:

            w_inc[k] = coeff[k] * path[p_idx[k]]

        for one path increment.
        """
        if self.coeff.size == 0:
            return np.empty(
                (0,),
                dtype=np.float64,
            )

        path_data = path._data.astype(
            np.float64,
            copy=False,
        )

        return (
            self.coeff
            * path_data[self.p_idx]
        )

    def eval2_weighted_add_into_data(
        self,
        out: np.ndarray,
        gd1: np.ndarray,
        gd2: np.ndarray,
        w_inc: np.ndarray,
    ) -> None:
        """
        Add the compiled polynomial evaluation into out using
        precomputed increment weights.
        """
        if w_inc.size == 0:
            return

        _eval2_weighted_add_into_kernel(
            out,
            self.out_idx,
            self.g1_idx,
            self.g2_idx,
            gd1,
            gd2,
            w_inc,
        )

    def eval2(
        self,
        gub1: TensorElement,
        gub2: TensorElement,
        path: TensorElement,
    ) -> TensorElement:
        """
        Evaluate the polynomial on two tensor-algebra arguments.
        """
        if (
            gub1.spec != self.spec_out
            or gub2.spec != self.spec_out
        ):
            raise ValueError("gub.spec mismatch")

        gd1 = gub1._data.astype(
            np.float64,
            copy=False,
        )
        gd2 = gub2._data.astype(
            np.float64,
            copy=False,
        )

        increment_weights = (
            self.precompute_inc_weights(path)
        )

        out = np.zeros(
            self.spec_out.total_dim,
            dtype=np.float64,
        )

        self.eval2_weighted_add_into_data(
            out,
            gd1,
            gd2,
            increment_weights,
        )

        return TensorElement(
            self.spec_out,
            out,
        )

    def eval(
        self,
        gub: TensorElement,
        path: TensorElement,
    ) -> TensorElement:
        """
        Evaluate the polynomial using the same tensor element
        for both tensor arguments.
        """
        return self.eval2(
            gub,
            gub,
            path,
        )


def compile_poly_numpy(
    gub_spec: TensorAlgebraSpec,
    path_spec: TensorAlgebraSpec,
    N: int,
    max_d: int,
) -> CompiledPolyNP:
    """
    Compile and compress the SDK polynomial.

    Terms sharing the same:

        (out_idx, g1_idx, g2_idx, p_idx)

    are combined during compilation. Terms whose accumulated
    coefficient is zero are removed.
    """
    pairing_cache = get_pairing_cache(
        N,
        max_d,
    )
    inverse_shuffle_cache = (
        get_inverse_shuffle_cache(
            gub_spec,
            path_spec,
        )
    )

    gub_words = [
        gub_spec.index_to_word(i)
        for i in range(gub_spec.total_dim)
    ]
    path_words = [
        path_spec.index_to_word(i)
        for i in range(path_spec.total_dim)
    ]

    TermKey = Tuple[int, int, int, int]
    term_coefficients: Dict[TermKey, float] = {}

    for out_i, word in enumerate(gub_words):
        if len(word) == 0:
            continue

        pre, post, mid = split_word(word)

        for path_word in path_words:
            mu = (mid,) + path_word

            if len(mu) > gub_spec.max_level:
                continue

            mu_idx = gub_spec.word_to_index(mu)

            inverse_shuffle_terms = (
                inverse_shuffle_cache.table[out_i][mu_idx]
            )

            for i1, i2 in inverse_shuffle_terms:
                left_word = gub_words[i1]
                right_word = gub_words[i2]

                left_terms = compiled_compute_vals_terms(
                    left_word,
                    len(pre),
                    gub_spec,
                    pairing_cache,
                )

                right_terms = compiled_compute_vals_terms(
                    right_word,
                    len(post),
                    gub_spec,
                    pairing_cache,
                    backward=True,
                )

                for left_idx, left_coefficient in left_terms:
                    for right_idx, right_coefficient in right_terms:
                        key = (
                            out_i,
                            left_idx,
                            right_idx,
                            mu_idx,
                        )

                        coefficient = (
                            -left_coefficient
                            * right_coefficient
                        )

                        term_coefficients[key] = (
                            term_coefficients.get(
                                key,
                                0.0,
                            )
                            + coefficient
                        )

    out_idx: List[int] = []
    g1_idx: List[int] = []
    g2_idx: List[int] = []
    p_idx: List[int] = []
    coeff: List[float] = []

    for key, coefficient in term_coefficients.items():
        if coefficient == 0.0:
            continue

        (
            output_index,
            left_index,
            right_index,
            path_index,
        ) = key

        out_idx.append(output_index)
        g1_idx.append(left_index)
        g2_idx.append(right_index)
        p_idx.append(path_index)
        coeff.append(coefficient)

    return CompiledPolyNP(
        spec_out=gub_spec,
        out_idx=np.asarray(
            out_idx,
            dtype=np.int64,
        ),
        g1_idx=np.asarray(
            g1_idx,
            dtype=np.int64,
        ),
        g2_idx=np.asarray(
            g2_idx,
            dtype=np.int64,
        ),
        p_idx=np.asarray(
            p_idx,
            dtype=np.int64,
        ),
        coeff=np.asarray(
            coeff,
            dtype=np.float64,
        ),
    )