import numpy as np
import pytest

from src.sdker.polynomial import compile_poly_numpy
from src.sdker.randomized_rde import RandomizedRDE
from src.sdker.reference_solver import SimpleSolver
from src.sdker.signatures import signature_increments_from_path
from src.sdker.solver import SDKSolverCompiled, SDKSolverRuntime
from src.sdker.tensor_algebra import TensorAlgebraSpec


def make_sdk_solver(path_dim, depth):
    solution_spec = TensorAlgebraSpec(
        dim=path_dim,
        max_level=depth,
    )
    tail_spec = TensorAlgebraSpec(
        dim=path_dim,
        max_level=depth - 1,
    )

    polynomial = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=depth,
        max_d=depth,
    )

    compiled = SDKSolverCompiled(
        poly=polynomial,
        gub_spec=solution_spec,
        max_d=depth,
    )

    return solution_spec, SDKSolverRuntime(compiled)


def compute_sdk_kernel(path, depth=3, block_size=1):
    path = np.asarray(path, dtype=float)

    expected_spec, solver = make_sdk_solver(
        path_dim=path.shape[1],
        depth=depth,
    )

    increment_spec, signature_increments = (
        signature_increments_from_path(
            path=path,
            M=block_size,
            depth=depth,
        )
    )

    assert increment_spec == expected_spec

    G = solver.compute(signature_increments)

    # The scalar Schwinger-Dyson kernel is the empty-word
    # coordinate of the interval [0,T].
    return G[0][-1][()]


def linear_path(displacement, n_steps):
    displacement = np.asarray(
        displacement,
        dtype=float,
    )

    times = np.linspace(0.0, 1.0, n_steps + 1)

    return times[:, None] * displacement[None, :]


def test_all_three_methods_equal_one_on_zero_path():
    path = np.zeros((6, 2))

    sdk_value = compute_sdk_kernel(
        path,
        depth=3,
        block_size=1,
    )

    reference_value = SimpleSolver(path).compute()[0, -1]

    randomized_value = RandomizedRDE(
        path_increments=np.zeros(2),
        matrix_dim=4,
        N_simul=100,
        rng=np.random.default_rng(123),
    ).compute()

    assert sdk_value == pytest.approx(
        1.0,
        rel=1e-13,
        abs=1e-13,
    )
    assert reference_value == pytest.approx(
        1.0,
        rel=1e-13,
        abs=1e-13,
    )
    assert randomized_value == pytest.approx(
        1.0 + 0.0j,
        rel=1e-13,
        abs=1e-13,
    )


def test_sdk_and_reference_converge_under_mesh_refinement():
    displacement = np.array([0.2])

    errors = []

    for n_steps in [5, 10, 20]:
        path = linear_path(
            displacement=displacement,
            n_steps=n_steps,
        )

        sdk_value = compute_sdk_kernel(
            path,
            depth=3,
            block_size=1,
        )

        reference_value = (
            SimpleSolver(path).compute()[0, -1]
        )

        errors.append(
            abs(sdk_value - reference_value)
        )

    # For this fixed smooth benchmark, the two discretizations
    # approach one another as the mesh is refined.
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]

    assert errors[-1] < 6e-3


@pytest.mark.parametrize(
    "depth",
    [1, 2, 3],
)
def test_sdk_reference_agreement_on_refined_linear_path(depth):
    path = linear_path(
        displacement=np.array([0.1]),
        n_steps=20,
    )

    sdk_value = compute_sdk_kernel(
        path,
        depth=depth,
        block_size=1,
    )

    reference_value = SimpleSolver(path).compute()[0, -1]

    # At this mesh size, all three truncation levels should already
    # give a reasonably accurate approximation for this small path.
    assert sdk_value == pytest.approx(
        reference_value,
        abs=2e-3,
    )


def test_reference_agrees_with_randomized_rde_on_linear_path():
    displacement = np.array([0.2])

    path = linear_path(
        displacement=displacement,
        n_steps=50,
    )

    reference_value = SimpleSolver(path).compute()[0, -1]

    randomized_value = RandomizedRDE(
        path_increments=displacement,
        matrix_dim=4,
        N_simul=2_000,
        rng=np.random.default_rng(2026),
    ).compute()

    # The theoretical target is real. The imaginary part of the
    # Monte Carlo estimator vanishes only in expectation.
    assert randomized_value.real == pytest.approx(
        reference_value,
        abs=4e-3,
    )

    assert randomized_value.imag == pytest.approx(
        0.0,
        abs=5e-3,
    )


def test_sdk_agrees_with_both_oracles_on_refined_linear_path():
    displacement = np.array([0.2])

    path = linear_path(
        displacement=displacement,
        n_steps=40,
    )

    sdk_value = compute_sdk_kernel(
        path,
        depth=3,
        block_size=1,
    )

    reference_value = SimpleSolver(path).compute()[0, -1]

    randomized_value = RandomizedRDE(
        path_increments=displacement,
        matrix_dim=4,
        N_simul=2_000,
        rng=np.random.default_rng(12345),
    ).compute()

    assert sdk_value == pytest.approx(
        reference_value,
        abs=5e-3,
    )

    assert randomized_value.real == pytest.approx(
        reference_value,
        abs=5e-3,
    )

    assert sdk_value == pytest.approx(
        randomized_value.real,
        abs=7e-3,
    )

    assert randomized_value.imag == pytest.approx(
        0.0,
        abs=5e-3,
    )