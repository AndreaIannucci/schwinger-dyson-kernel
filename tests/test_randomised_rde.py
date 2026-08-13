import numpy as np
import pytest
from scipy.linalg import expm

from src.sdker.randomized_rde import (
    RandomizedRDE,
    _one_rde_simulation,
    random_u_lie_algebra,
)


@pytest.mark.parametrize("matrix_dim", [1, 2, 5, 10])
def test_random_lie_algebra_element_has_correct_shape(matrix_dim):
    rng = np.random.default_rng(123)

    A = random_u_lie_algebra(matrix_dim, rng)

    assert A.shape == (matrix_dim, matrix_dim)
    assert np.iscomplexobj(A)


@pytest.mark.parametrize("matrix_dim", [1, 2, 5, 10])
def test_random_lie_algebra_element_is_anti_hermitian(matrix_dim):
    rng = np.random.default_rng(456)

    A = random_u_lie_algebra(matrix_dim, rng)

    np.testing.assert_allclose(
        A.conj().T,
        -A,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("matrix_dim", [1, 2, 5])
def test_exponential_is_unitary(matrix_dim):
    rng = np.random.default_rng(789)

    A = random_u_lie_algebra(matrix_dim, rng)
    U = expm(A)

    identity = np.eye(matrix_dim)

    np.testing.assert_allclose(
        U.conj().T @ U,
        identity,
        rtol=1e-12,
        atol=1e-12,
    )


def test_random_lie_algebra_is_reproducible():
    first = random_u_lie_algebra(
        4,
        np.random.default_rng(123),
    )
    second = random_u_lie_algebra(
        4,
        np.random.default_rng(123),
    )

    np.testing.assert_array_equal(first, second)


def test_one_simulation_is_reproducible_for_fixed_seed():
    path_increment = np.array([0.2, -0.1, 0.3])

    first = _one_rde_simulation(
        seed=123,
        path_increments=path_increment,
        N=4,
    )
    second = _one_rde_simulation(
        seed=123,
        path_increments=path_increment,
        N=4,
    )

    assert first == second


@pytest.mark.parametrize(
    ("path_dim", "matrix_dim"),
    [
        (1, 1),
        (2, 3),
        (5, 7),
    ],
)
def test_zero_increment_has_trace_equal_to_matrix_dimension(
    path_dim,
    matrix_dim,
):
    trace = _one_rde_simulation(
        seed=123,
        path_increments=np.zeros(path_dim),
        N=matrix_dim,
    )

    assert trace == pytest.approx(
        complex(matrix_dim),
        rel=1e-13,
        abs=1e-13,
    )


def test_one_simulation_trace_is_bounded():
    matrix_dim = 5

    trace = _one_rde_simulation(
        seed=123,
        path_increments=np.array([0.2, -0.3]),
        N=matrix_dim,
    )

    # The trace of a unitary N x N matrix has modulus at most N.
    assert abs(trace) <= matrix_dim + 1e-12


@pytest.mark.parametrize(
    ("path_dim", "matrix_dim", "n_simulations"),
    [
        (1, 1, 1),
        (2, 3, 5),
        (4, 5, 10),
    ],
)
def test_compute_zero_increment_returns_one(
    path_dim,
    matrix_dim,
    n_simulations,
):
    estimator = RandomizedRDE(
        path_increments=np.zeros(path_dim),
        matrix_dim=matrix_dim,
        N_simul=n_simulations,
    )

    result = estimator.compute()

    assert result == pytest.approx(
        1.0 + 0.0j,
        rel=1e-13,
        abs=1e-13,
    )


def test_compute_returns_complex_number():
    estimator = RandomizedRDE(
        path_increments=np.array([0.2, -0.1]),
        matrix_dim=4,
        N_simul=5,
    )

    result = estimator.compute()

    assert isinstance(result, complex)


def test_compute_normalized_trace_is_bounded():
    estimator = RandomizedRDE(
        path_increments=np.array([0.2, -0.1, 0.3]),
        matrix_dim=5,
        N_simul=20,
        rng=np.random.default_rng(123),
    )

    result = estimator.compute()

    assert abs(result) <= 1.0 + 1e-12


def test_compute_is_reproducible_with_equal_rng_seeds():
    first = RandomizedRDE(
        path_increments=np.array([0.2, -0.1]),
        matrix_dim=4,
        N_simul=10,
        rng=np.random.default_rng(123),
    ).compute()

    second = RandomizedRDE(
        path_increments=np.array([0.2, -0.1]),
        matrix_dim=4,
        N_simul=10,
        rng=np.random.default_rng(123),
    ).compute()

    assert first == second


def test_compute_parallel_zero_increment_returns_one():
    estimator = RandomizedRDE(
        path_increments=np.zeros(3),
        matrix_dim=4,
        N_simul=10,
        rng=np.random.default_rng(123),
    )

    result = estimator.compute_parallel(n_workers=2)

    assert result == pytest.approx(
        1.0 + 0.0j,
        rel=1e-13,
        abs=1e-13,
    )


def test_compute_parallel_is_reproducible_with_equal_rng_seeds():
    first = RandomizedRDE(
        path_increments=np.array([0.2, -0.1]),
        matrix_dim=4,
        N_simul=10,
        rng=np.random.default_rng(123),
    ).compute_parallel(n_workers=2)

    second = RandomizedRDE(
        path_increments=np.array([0.2, -0.1]),
        matrix_dim=4,
        N_simul=10,
        rng=np.random.default_rng(123),
    ).compute_parallel(n_workers=2)

    assert first == second