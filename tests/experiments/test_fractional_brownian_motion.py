import numpy as np
import pytest

from experiments.common.fractional_brownian_motion import (
    fbm_davies_harte,
)


def test_output_shape():
    path = fbm_davies_harte(
        dim=3,
        n_steps=16,
        H=0.7,
        rng=np.random.default_rng(123),
    )

    assert path.shape == (17, 3)


def test_path_starts_at_zero():
    path = fbm_davies_harte(
        dim=4,
        n_steps=16,
        H=0.7,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(
        path[0],
        np.zeros(4),
    )


def test_reproducible_with_fixed_seed():
    first = fbm_davies_harte(
        dim=3,
        n_steps=16,
        H=0.7,
        rng=np.random.default_rng(123),
    )

    second = fbm_davies_harte(
        dim=3,
        n_steps=16,
        H=0.7,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(first, second)


def test_default_rng_is_reproducible():
    first = fbm_davies_harte(
        dim=3,
        n_steps=16,
        H=0.7,
    )

    second = fbm_davies_harte(
        dim=3,
        n_steps=16,
        H=0.7,
    )

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("H", [0.0, 1.0, -0.1, 1.1])
def test_rejects_invalid_hurst_parameter(H):
    with pytest.raises(ValueError):
        fbm_davies_harte(
            dim=2,
            n_steps=16,
            H=H,
        )


@pytest.mark.parametrize("dim", [0, -1])
def test_rejects_invalid_dimension(dim):
    with pytest.raises(ValueError):
        fbm_davies_harte(
            dim=dim,
            n_steps=16,
            H=0.7,
        )


@pytest.mark.parametrize("n_steps", [0, -1])
def test_rejects_invalid_number_of_steps(n_steps):
    with pytest.raises(ValueError):
        fbm_davies_harte(
            dim=2,
            n_steps=n_steps,
            H=0.7,
        )


def test_brownian_increment_variance():
    dim = 5_000
    n_steps = 16
    H = 0.5

    path = fbm_davies_harte(
        dim=dim,
        n_steps=n_steps,
        H=H,
        rng=np.random.default_rng(123),
    )

    increments = np.diff(path, axis=0)
    first_increment = increments[0]

    empirical_variance = np.var(
        first_increment,
        ddof=1,
    )

    expected_variance = 1.0 / n_steps

    assert empirical_variance == pytest.approx(
        expected_variance,
        rel=0.08,
    )


def test_fractional_increment_variance():
    dim = 5_000
    n_steps = 16
    H = 0.75

    path = fbm_davies_harte(
        dim=dim,
        n_steps=n_steps,
        H=H,
        rng=np.random.default_rng(456),
    )

    increments = np.diff(path, axis=0)

    empirical_variance = np.var(
        increments[0],
        ddof=1,
    )

    expected_variance = (
        1.0 / n_steps
    ) ** (2.0 * H)

    assert empirical_variance == pytest.approx(
        expected_variance,
        rel=0.08,
    )


def test_fractional_increment_lag_one_covariance():
    dim = 6_000
    n_steps = 16
    H = 0.75

    path = fbm_davies_harte(
        dim=dim,
        n_steps=n_steps,
        H=H,
        rng=np.random.default_rng(789),
    )

    increments = np.diff(path, axis=0)

    empirical_covariance = np.cov(
        increments[0],
        increments[1],
        ddof=1,
    )[0, 1]

    unit_lag_covariance = 0.5 * (
        2.0 ** (2.0 * H) - 2.0
    )

    expected_covariance = (
        (1.0 / n_steps) ** (2.0 * H)
        * unit_lag_covariance
    )

    assert empirical_covariance == pytest.approx(
        expected_covariance,
        rel=0.12,
    )


def test_endpoint_variance_is_one():
    dim = 5_000

    path = fbm_davies_harte(
        dim=dim,
        n_steps=32,
        H=0.7,
        rng=np.random.default_rng(987),
    )

    empirical_variance = np.var(
        path[-1],
        ddof=1,
    )

    # Fractional Brownian motion on [0,1] satisfies Var(B_1) = 1.
    assert empirical_variance == pytest.approx(
        1.0,
        rel=0.08,
    )


def test_brownian_increments_are_uncorrelated():
    dim = 6_000
    n_steps = 16

    path = fbm_davies_harte(
        dim=dim,
        n_steps=n_steps,
        H=0.5,
        rng=np.random.default_rng(2468),
    )

    increments = np.diff(path, axis=0)

    empirical_covariance = np.cov(
        increments[0],
        increments[1],
        ddof=1,
    )[0, 1]

    expected_variance = 1.0 / n_steps

    assert abs(empirical_covariance) < (
        0.06 * expected_variance
    )