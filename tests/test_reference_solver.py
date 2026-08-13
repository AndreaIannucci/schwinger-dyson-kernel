import numpy as np
import pytest

from sdker.reference_solver import SimpleSolver


def direct_quartic_solver(gamma):
    """
    Direct implementation of the defining discrete recurrence.

    This is deliberately slow and is retained only as a test oracle.
    """
    gamma = np.asarray(gamma, dtype=float)

    n = gamma.shape[0] - 1
    dim = gamma.shape[1]

    dgamma = gamma[1:] - gamma[:-1]

    K = np.zeros((n + 1, n + 1), dtype=float)
    np.fill_diagonal(K, 1.0)

    for length in range(1, n + 1):
        for a in range(n + 1 - length):
            b = a + length
            value = 1.0

            for coordinate in range(dim):
                for m in range(a, b):
                    inner = 0.0

                    for k in range(a, m):
                        inner += (
                            K[a, k]
                            * K[k, m]
                            * dgamma[k, coordinate]
                        )

                    value -= (
                        inner
                        * dgamma[m, coordinate]
                    )

            K[a, b] = value

    return K


def test_rejects_non_matrix_input():
    with pytest.raises(ValueError):
        SimpleSolver(np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError):
        SimpleSolver(np.zeros((2, 2, 2)))


def test_input_is_converted_to_float_array():
    solver = SimpleSolver(
        [
            [0, 0],
            [1, 2],
        ]
    )

    assert isinstance(solver.gamma, np.ndarray)
    assert np.issubdtype(solver.gamma.dtype, np.floating)


def test_single_point_path():
    gamma = np.array(
        [
            [2.0, -1.0],
        ]
    )

    K = SimpleSolver(gamma).compute()

    np.testing.assert_array_equal(
        K,
        np.array([[1.0]]),
    )


def test_output_shape():
    gamma = np.zeros((7, 3))

    K = SimpleSolver(gamma).compute()

    assert K.shape == (7, 7)


def test_diagonal_is_one():
    rng = np.random.default_rng(123)
    gamma = rng.normal(size=(8, 3))

    K = SimpleSolver(gamma).compute()

    np.testing.assert_array_equal(
        np.diag(K),
        np.ones(8),
    )


def test_lower_triangle_is_zero():
    rng = np.random.default_rng(456)
    gamma = rng.normal(size=(8, 2))

    K = SimpleSolver(gamma).compute()

    np.testing.assert_array_equal(
        np.tril(K, k=-1),
        np.zeros_like(K),
    )


def test_constant_path_produces_unit_upper_triangle():
    gamma = np.full((6, 3), 2.5)

    K = SimpleSolver(gamma).compute()

    expected = np.triu(np.ones((6, 6)))

    np.testing.assert_array_equal(K, expected)


def test_one_increment_path():
    gamma = np.array(
        [
            [0.0, 0.0],
            [3.0, -2.0],
        ]
    )

    K = SimpleSolver(gamma).compute()

    expected = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(K, expected)


def test_two_increment_one_dimensional_path():
    gamma = np.array(
        [
            [0.0],
            [0.2],
            [0.5],
        ]
    )

    K = SimpleSolver(gamma).compute()

    # K[0,2] = 1 - Δgamma_0 Δgamma_1
    expected_final = 1.0 - 0.2 * 0.3

    assert K[0, 2] == pytest.approx(expected_final)


def test_two_increment_multidimensional_path():
    gamma = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [4.0, 1.0],
        ]
    )

    first_increment = gamma[1] - gamma[0]
    second_increment = gamma[2] - gamma[1]

    K = SimpleSolver(gamma).compute()

    expected_final = 1.0 - np.dot(
        first_increment,
        second_increment,
    )

    assert K[0, 2] == pytest.approx(expected_final)


@pytest.mark.parametrize(
    ("n_increments", "dim"),
    [
        (1, 1),
        (2, 1),
        (4, 2),
        (6, 3),
        (8, 2),
    ],
)
def test_solver_agrees_with_direct_quartic_implementation(
    n_increments,
    dim,
):
    rng = np.random.default_rng(
        100 * n_increments + dim
    )

    increments = rng.normal(
        scale=0.05,
        size=(n_increments, dim),
    )

    gamma = np.vstack(
        [
            np.zeros((1, dim)),
            np.cumsum(increments, axis=0),
        ]
    )

    expected = direct_quartic_solver(gamma)
    actual = SimpleSolver(gamma).compute()

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-13,
        atol=1e-13,
    )


def test_solver_satisfies_right_endpoint_recurrence():
    rng = np.random.default_rng(789)

    increments = rng.normal(
        scale=0.1,
        size=(7, 3),
    )

    gamma = np.vstack(
        [
            np.zeros((1, 3)),
            np.cumsum(increments, axis=0),
        ]
    )

    K = SimpleSolver(gamma).compute()
    dgamma = np.diff(gamma, axis=0)

    n = gamma.shape[0] - 1

    for b in range(1, n + 1):
        for a in range(b):
            correction = 0.0

            for k in range(a, b - 1):
                correction += (
                    K[a, k]
                    * K[k, b - 1]
                    * np.dot(dgamma[k], dgamma[b - 1])
                )

            expected = K[a, b - 1] - correction

            assert K[a, b] == pytest.approx(
                expected,
                rel=1e-12,
                abs=1e-12,
            )


def test_compute_is_repeatable():
    rng = np.random.default_rng(987)
    gamma = rng.normal(size=(6, 2))

    solver = SimpleSolver(gamma)

    first = solver.compute()
    second = solver.compute()

    np.testing.assert_array_equal(first, second)


def test_compute_does_not_modify_path():
    rng = np.random.default_rng(654)
    gamma = rng.normal(size=(6, 2))
    original = gamma.copy()

    SimpleSolver(gamma).compute()

    np.testing.assert_array_equal(gamma, original)