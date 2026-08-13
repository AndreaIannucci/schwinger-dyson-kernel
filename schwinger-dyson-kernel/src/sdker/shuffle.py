from typing import Dict, List, Tuple, Set, Optional

def inverse_shuffle_masks(n):
    """
    Yield all 0/1 masks of length n (inverse shuffle coproduct).
    0 = goes to first word
    1 = goes to second word
    """
    for mask in range(1 << n):
        yield mask

def mask_to_bits(mask, n):
    return [(mask >> i) & 1 for i in range(n)]

_inverse_shuffle_masks_cache: Dict[int, Tuple[int, ...]] = {}


def inverse_shuffle_masks_cached(n: int) -> Tuple[int, ...]:
    """
    Return all 0/1 masks of length n, cached.
    0 -> goes to first word
    1 -> goes to second word
    """
    if n not in _inverse_shuffle_masks_cache:
        _inverse_shuffle_masks_cache[n] = tuple(range(1 << n))
    return _inverse_shuffle_masks_cache[n]


def inverse_shuffle_word(word: Tuple[int, ...]):
    """
    Yield all inverse shuffles (unshuffles) of a word.

    Input:
        word = (a1, ..., an)

    Output:
        (w1, w2) such that:
          - w1, w2 are order-preserving subsequences
          - w1 ∪ w2 = word
          - corresponds to inverse shuffle coproduct
    """
    n = len(word)

    for mask in inverse_shuffle_masks_cached(n):
        w1 = []
        w2 = []

        # LSB-first convention, consistent with mask generation
        for i in range(n):
            if (mask >> i) & 1 == 0:
                w1.append(word[i])
            else:
                w2.append(word[i])

        yield tuple(w1), tuple(w2)




