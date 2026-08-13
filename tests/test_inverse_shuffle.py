from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from src.sdker.inverse_shuffle import (
    build_inverse_shuffle_cache,
    get_inverse_shuffle_cache,
    split_word,
)
from src.sdker.shuffle import inverse_shuffle_word
from src.sdker.tensor_algebra import TensorAlgebraSpec


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ((1,), ((), (), 1)),
        ((1, 2), ((1,), (), 2)),
        ((1, 2, 3), ((1,), (3,), 2)),
        ((1, 2, 3, 4), ((1, 2), (4,), 3)),
        (
            (1, 2, 3, 4, 5),
            ((1, 2), (4, 5), 3),
        ),
    ],
)
def test_split_word(word, expected):
    assert split_word(word) == expected


def test_split_word_rejects_empty_word():
    with pytest.raises(ValueError):
        split_word(())


def test_cache_has_expected_dimensions():
    spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    cache = build_inverse_shuffle_cache(spec, tail_spec)

    assert len(cache.table) == spec.total_dim

    for row in cache.table:
        assert len(row) == spec.total_dim


def test_empty_word_row_contains_no_terms():
    spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    cache = build_inverse_shuffle_cache(spec, tail_spec)
    empty_index = spec.word_to_index(())

    assert all(
        bucket == ()
        for bucket in cache.table[empty_index]
    )


def test_explicit_one_dimensional_cache():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=1, max_level=1)

    cache = build_inverse_shuffle_cache(spec, tail_spec)

    empty = spec.word_to_index(())
    one = spec.word_to_index((1,))
    two = spec.word_to_index((1, 1))

    # Output word w = (1), with u = ().
    assert cache.table[one][one] == (
        (empty, empty),
    )

    # Output word w = (1), with u = (1).
    assert cache.table[one][two] == (
        (one, empty),
        (empty, one),
    )

    # Output word w = (1,1), with u = ().
    assert cache.table[two][one] == (
        (one, empty),
    )

    # Output word w = (1,1), with u = (1).
    assert cache.table[two][two] == (
        (two, empty),
        (one, one),
    )


def direct_cache_bucket(spec, tail_spec, word, mu):
    """
    Independently reconstruct one cache bucket in word form.
    """
    if len(word) == 0:
        return ()

    pre, post, mid = split_word(word)

    if len(mu) == 0 or mu[0] != mid:
        return ()

    u = mu[1:]

    if len(u) > tail_spec.max_level:
        return ()

    # Make sure u is actually a word in tail_spec.
    try:
        tail_spec.word_to_index(u)
    except ValueError:
        return ()

    terms = []

    for u1, u2 in inverse_shuffle_word(u):
        left_word = pre + u1
        right_word = post + u2

        if (
            len(left_word) <= spec.max_level
            and len(right_word) <= spec.max_level
        ):
            terms.append((left_word, right_word))

    return tuple(terms)


@pytest.mark.parametrize(
    ("dim", "max_level"),
    [
        (1, 1),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 2),
    ],
)
def test_cache_agrees_with_direct_word_construction(
    dim,
    max_level,
):
    spec = TensorAlgebraSpec(
        dim=dim,
        max_level=max_level,
    )
    tail_spec = TensorAlgebraSpec(
        dim=dim,
        max_level=max_level - 1,
    )

    cache = build_inverse_shuffle_cache(spec, tail_spec)

    for word in spec.iter_words():
        word_index = spec.word_to_index(word)

        for mu in spec.iter_words():
            mu_index = spec.word_to_index(mu)

            cached_terms = tuple(
                (
                    spec.index_to_word(left_index),
                    spec.index_to_word(right_index),
                )
                for left_index, right_index
                in cache.table[word_index][mu_index]
            )

            expected_terms = direct_cache_bucket(
                spec=spec,
                tail_spec=tail_spec,
                word=word,
                mu=mu,
            )

            assert cached_terms == expected_terms


def test_cached_indices_are_in_range():
    spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    cache = build_inverse_shuffle_cache(spec, tail_spec)

    for row in cache.table:
        for bucket in row:
            for left_index, right_index in bucket:
                assert 0 <= left_index < spec.total_dim
                assert 0 <= right_index < spec.total_dim


def test_cache_preserves_inverse_shuffle_multiplicity():
    spec = TensorAlgebraSpec(dim=1, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=1, max_level=2)

    cache = build_inverse_shuffle_cache(spec, tail_spec)

    word_index = spec.word_to_index((1,))
    mu_index = spec.word_to_index((1, 1, 1))

    empty = spec.word_to_index(())
    one = spec.word_to_index((1,))
    two = spec.word_to_index((1, 1))

    multiplicities = Counter(
        cache.table[word_index][mu_index]
    )

    assert multiplicities[(two, empty)] == 1
    assert multiplicities[(one, one)] == 2
    assert multiplicities[(empty, two)] == 1


def test_get_inverse_shuffle_cache_reuses_object():
    spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    first = get_inverse_shuffle_cache(spec, tail_spec)
    second = get_inverse_shuffle_cache(spec, tail_spec)

    assert first is second


def test_different_specs_produce_different_cache_objects():
    spec1 = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec1 = TensorAlgebraSpec(dim=2, max_level=1)

    spec2 = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec2 = TensorAlgebraSpec(dim=2, max_level=2)

    first = get_inverse_shuffle_cache(spec1, tail_spec1)
    second = get_inverse_shuffle_cache(spec2, tail_spec2)

    assert first is not second


def test_cache_is_immutable():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    cache = get_inverse_shuffle_cache(spec, tail_spec)

    with pytest.raises(FrozenInstanceError):
        cache.spec = tail_spec