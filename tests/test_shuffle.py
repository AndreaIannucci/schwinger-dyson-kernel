from collections import Counter

import pytest

from sdker.shuffle import (
    inverse_shuffle_masks,
    inverse_shuffle_masks_cached,
    inverse_shuffle_word,
    mask_to_bits,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 2, 3]),
        (3, list(range(8))),
    ],
)
def test_inverse_shuffle_masks(n, expected):
    assert list(inverse_shuffle_masks(n)) == expected


@pytest.mark.parametrize(
    ("mask", "n", "expected"),
    [
        (0, 3, [0, 0, 0]),
        (1, 3, [1, 0, 0]),
        (2, 3, [0, 1, 0]),
        (3, 3, [1, 1, 0]),
        (5, 3, [1, 0, 1]),
        (7, 3, [1, 1, 1]),
    ],
)
def test_mask_to_bits(mask, n, expected):
    assert mask_to_bits(mask, n) == expected


def test_cached_masks_agree_with_direct_generator():
    for n in range(8):
        expected = tuple(inverse_shuffle_masks(n))
        actual = inverse_shuffle_masks_cached(n)

        assert actual == expected


def test_masks_are_actually_cached():
    first = inverse_shuffle_masks_cached(5)
    second = inverse_shuffle_masks_cached(5)

    assert first is second


def test_inverse_shuffle_of_empty_word():
    assert list(inverse_shuffle_word(())) == [
        ((), ()),
    ]


def test_inverse_shuffle_of_one_letter_word():
    assert list(inverse_shuffle_word((1,))) == [
        ((1,), ()),
        ((), (1,)),
    ]


def test_inverse_shuffle_of_two_letter_word():
    assert list(inverse_shuffle_word((1, 2))) == [
        ((1, 2), ()),
        ((2,), (1,)),
        ((1,), (2,)),
        ((), (1, 2)),
    ]


def test_inverse_shuffle_of_three_letter_word():
    assert list(inverse_shuffle_word((1, 2, 3))) == [
        ((1, 2, 3), ()),
        ((2, 3), (1,)),
        ((1, 3), (2,)),
        ((3,), (1, 2)),
        ((1, 2), (3,)),
        ((2,), (1, 3)),
        ((1,), (2, 3)),
        ((), (1, 2, 3)),
    ]


@pytest.mark.parametrize("n", range(9))
def test_inverse_shuffle_has_two_to_the_n_terms(n):
    word = tuple(range(1, n + 1))

    shuffles = list(inverse_shuffle_word(word))

    assert len(shuffles) == 2**n


def is_subsequence(subsequence, word):
    iterator = iter(word)

    return all(
        any(letter == candidate for candidate in iterator)
        for letter in subsequence
    )


@pytest.mark.parametrize("n", range(1, 8))
def test_each_inverse_shuffle_preserves_order_and_positions(n):
    # Distinct letters allow us to recover their original positions.
    word = tuple(range(1, n + 1))

    for left, right in inverse_shuffle_word(word):
        assert is_subsequence(left, word)
        assert is_subsequence(right, word)

        combined_positions = sorted(left + right)

        assert combined_positions == list(word)
        assert set(left).isdisjoint(right)


def test_repeated_letters_retain_multiplicity():
    shuffles = list(inverse_shuffle_word((1, 1)))

    # There are four position-based inverse shuffles even though the
    # middle two produce the same pair of words.
    assert len(shuffles) == 4

    multiplicities = Counter(shuffles)

    assert multiplicities[((1, 1), ())] == 1
    assert multiplicities[((1,), (1,))] == 2
    assert multiplicities[((), (1, 1))] == 1


def test_every_letter_goes_to_exactly_one_output_word():
    word = (1, 2, 3, 4, 5)

    for left, right in inverse_shuffle_word(word):
        assert len(left) + len(right) == len(word)
        assert sorted(left + right) == list(word)