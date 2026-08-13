from dataclasses import FrozenInstanceError

import pytest

from src.sdker.combinatorics import (
    get_pairing_cache,
    saturated_pairings_for_d,
)


def test_level_zero_configuration():
    cache = get_pairing_cache(N=4, max_d=4)

    assert cache.levels[0] == (
        ((), ()),
    )


def test_level_one_configuration():
    cache = get_pairing_cache(N=4, max_d=4)

    assert cache.levels[1] == (
        ((), (1,)),
    )


def test_level_two_configurations():
    cache = get_pairing_cache(N=4, max_d=4)

    assert cache.levels[2] == (
        ((), (1, 2)),
        (((1, 2),), ()),
    )


def test_level_three_configurations():
    cache = get_pairing_cache(N=4, max_d=4)

    assert cache.levels[3] == (
        ((), (1, 2, 3)),
        (((2, 3),), (1,)),
        (((1, 2),), (3,)),
    )


def test_level_four_configurations():
    cache = get_pairing_cache(N=4, max_d=4)

    assert cache.levels[4] == (
        ((), (1, 2, 3, 4)),
        (((3, 4),), (1, 2)),
        (((2, 3),), (1, 4)),
        (((2, 3), (1, 4)), ()),
        (((1, 2),), (3, 4)),
        (((1, 2), (3, 4)), ()),
    )


@pytest.mark.parametrize("N", range(8))
def test_every_position_is_used_exactly_once(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for n, configurations in enumerate(cache.levels):
        expected_positions = list(range(1, n + 1))

        for pairs, unmatched in configurations:
            paired_positions = [
                position
                for pair in pairs
                for position in pair
            ]

            all_positions = paired_positions + list(unmatched)

            assert sorted(all_positions) == expected_positions
            assert len(all_positions) == len(set(all_positions))


@pytest.mark.parametrize("N", range(8))
def test_every_pair_is_ordered(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for configurations in cache.levels:
        for pairs, _ in configurations:
            assert all(a < b for a, b in pairs)


def pairs_cross(first_pair, second_pair):
    a, b = first_pair
    c, d = second_pair

    return (
        a < c < b < d
        or c < a < d < b
    )


@pytest.mark.parametrize("N", range(8))
def test_pairings_are_non_crossing(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for configurations in cache.levels:
        for pairs, _ in configurations:
            for first_index, first_pair in enumerate(pairs):
                for second_pair in pairs[first_index + 1 :]:
                    assert not pairs_cross(first_pair, second_pair)


@pytest.mark.parametrize("N", range(8))
def test_levels_contain_no_duplicate_configurations(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for configurations in cache.levels:
        assert len(configurations) == len(set(configurations))


@pytest.mark.parametrize("N", range(8))
def test_first_close_is_correct(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for n, configurations in enumerate(cache.levels):
        assert len(cache.first_close[n]) == len(configurations)

        for configuration, first_close in zip(
            configurations,
            cache.first_close[n],
        ):
            pairs, _ = configuration

            expected = min(
                (right_endpoint for _, right_endpoint in pairs),
                default=N + 1,
            )

            assert first_close == expected


def test_d_zero_allows_every_configuration():
    cache = get_pairing_cache(N=6, max_d=6)

    for n in range(7):
        assert cache.table_dn[n][0] == cache.levels[n]


@pytest.mark.parametrize(
    ("d", "expected_count"),
    [
        (0, 6),
        (1, 6),
        (2, 4),
        (3, 2),
        (4, 1),
    ],
)
def test_d_restriction_for_level_four(d, expected_count):
    cache = get_pairing_cache(N=4, max_d=4)

    configurations = cache.table_dn[4][d]

    assert len(configurations) == expected_count

    for pairs, _ in configurations:
        assert all(
            not (a <= d and b <= d)
            for a, b in pairs
        )


@pytest.mark.parametrize("N", range(1, 8))
def test_every_d_table_entry_satisfies_restriction(N):
    cache = get_pairing_cache(N=N, max_d=N)

    for n in range(N + 1):
        for d in range(N + 1):
            for pairs, _ in cache.table_dn[n][d]:
                assert all(
                    not (a <= d and b <= d)
                    for a, b in pairs
                )


@pytest.mark.parametrize(
    ("n", "d"),
    [
        (1, 2),
        (2, 3),
        (2, 5),
        (4, 6),
    ],
)
def test_d_greater_than_n_returns_trivial_configuration(n, d):
    configurations = saturated_pairings_for_d(
        n=n,
        d=d,
        max_d=d,
    )

    assert configurations == (
        ((), tuple(range(1, n + 1))),
    )


def test_empty_level_is_trivial_for_every_d():
    cache = get_pairing_cache(N=0, max_d=5)

    for d in range(6):
        assert cache.table_dn[0][d] == (
            ((), ()),
        )


@pytest.mark.parametrize(
    ("n", "d", "max_d"),
    [
        (0, 0, 0),
        (2, 0, 4),
        (3, 2, 4),
        (4, 4, 4),
    ],
)
def test_convenience_function_agrees_with_cache(n, d, max_d):
    expected = get_pairing_cache(
        N=n,
        max_d=max_d,
    ).table_dn[n][d]

    actual = saturated_pairings_for_d(
        n=n,
        d=d,
        max_d=max_d,
    )

    assert actual == expected


def test_pairing_cache_is_reused():
    first = get_pairing_cache(N=6, max_d=8)
    second = get_pairing_cache(N=6, max_d=8)

    assert first is second


def test_different_parameters_produce_different_caches():
    first = get_pairing_cache(N=5, max_d=5)
    second = get_pairing_cache(N=6, max_d=6)

    assert first is not second


def test_cache_is_immutable():
    cache = get_pairing_cache(N=4, max_d=4)

    with pytest.raises(FrozenInstanceError):
        cache.N = 10


@pytest.mark.parametrize(
    ("N", "max_d"),
    [
        (-1, 0),
        (-2, 3),
        (3, -1),
    ],
)
def test_get_pairing_cache_rejects_invalid_parameters(N, max_d):
    with pytest.raises(ValueError):
        get_pairing_cache(N=N, max_d=max_d)


def test_saturated_pairings_rejects_negative_n():
    with pytest.raises(ValueError):
        saturated_pairings_for_d(n=-1, d=0)


@pytest.mark.parametrize("d", [-1, 4])
def test_saturated_pairings_rejects_d_outside_range(d):
    with pytest.raises(ValueError):
        saturated_pairings_for_d(
            n=3,
            d=d,
            max_d=3,
        )