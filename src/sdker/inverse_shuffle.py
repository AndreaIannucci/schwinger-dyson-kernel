from .tensor_algebra import TensorAlgebraSpec
from .shuffle import inverse_shuffle_word
from dataclasses import dataclass
from typing import Tuple, Dict, List


def split_word(word: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """
    Split a word into (pre, post, mid), where mid is the central letter.
    For len(word) == 1, returns ((), (), word[0]).
    """
    n = len(word)
    if n == 0:
        raise ValueError("Cannot split empty word")
    if n == 1:
        return (), (), word[0]

    mid_idx = n // 2
    pre_word = word[:mid_idx]
    post_word = word[mid_idx + 1:]
    return pre_word, post_word, word[mid_idx]


@dataclass(frozen=True)
class InverseShuffleCache:
    """
    Immutable cache for inverse shuffles.

    table[w_idx][mu_idx] = tuple of (i1_idx, i2_idx)

    where:
      - w_idx   : flat index of original word w in spec
      - mu_idx  : flat index of (mid,) + u in spec
      - i1_idx  : flat index of pre + u1 in spec
      - i2_idx  : flat index of post + u2 in spec
    """
    spec: "TensorAlgebraSpec"
    spec2: "TensorAlgebraSpec"
    table: Tuple[Tuple[Tuple[Tuple[int, int], ...], ...], ...]

def build_inverse_shuffle_cache(
    spec: "TensorAlgebraSpec",
    spec2: "TensorAlgebraSpec"
) -> InverseShuffleCache:

    # table[w_idx][mu_idx] = list of (i1_idx, i2_idx)
    buckets: List[List[List[Tuple[int, int]]]] = [
        [ [] for _ in range(spec.total_dim) ]
        for _ in range(spec.total_dim)
    ]

    # precompute u's from spec2
    words2 = [u for u in spec2.iter_words()]

    for w in spec.iter_words():
        if len(w) == 0:
            continue

        w_idx = spec.word_to_index(w)
        pre, post, mid = split_word(w)

        for u in words2:

            mu = (mid,) + u

            # truncation on the indexing word
            if len(mu) > spec.max_level:
                continue

            mu_idx = spec.word_to_index(mu)

            for u1, u2 in inverse_shuffle_word(u):

                left_word  = pre  + u1
                right_word = post + u2

                # truncation in T^N
                if (
                    len(left_word)  > spec.max_level or
                    len(right_word) > spec.max_level
                ):
                    continue

                i1 = spec.word_to_index(left_word)
                i2 = spec.word_to_index(right_word)

                buckets[w_idx][mu_idx].append((i1, i2))

    frozen_table = tuple(
        tuple(tuple(bucket) for bucket in row)
        for row in buckets
    )

    return InverseShuffleCache(spec, spec2, frozen_table)

_inverse_shuffle_cache: Dict[
    Tuple["TensorAlgebraSpec", "TensorAlgebraSpec"],
    InverseShuffleCache
] = {}


def get_inverse_shuffle_cache(
    spec: "TensorAlgebraSpec",
    spec2: "TensorAlgebraSpec"
) -> InverseShuffleCache:
    """
    Get (or build) inverse shuffle cache for (spec, spec2).
    """
    key = (spec, spec2)
    if key not in _inverse_shuffle_cache:
        _inverse_shuffle_cache[key] = build_inverse_shuffle_cache(spec, spec2)
    return _inverse_shuffle_cache[key]



