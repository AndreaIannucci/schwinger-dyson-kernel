import numpy as np
import pytest

from src.sdker.polynomial import compile_poly_numpy
from src.sdker.solver import SDKSolverCompiled, SDKSolverRuntime
from src.sdker.tensor_algebra import TensorAlgebraSpec, TensorElement


def make_compiled_solver(dim, depth):
    solution_spec = TensorAlgebraSpec(
        dim=dim,
        max_level=depth,
    )
    tail_spec = TensorAlgebraSpec(
        dim=dim,
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

    return solution_spec, compiled, SDKSolverRuntime(compiled)


def zero_signature(spec):
    return TensorElement.eye(spec)


def straight_line_signature(spec, displacement):
    """
    Exact truncated signature of a straight-line increment.
    """
    displacement = np.asarray(displacement, dtype=float)
    signature = TensorElement.eye(spec)

    tensor_power = np.array(1.0)
    factorial = 1.0

    for level in range(1, spec.max_level + 1):
        tensor_power = np.multiply.outer(
            tensor_power,
            displacement,
        )
        factorial *= level

        signature.level_view(level)[:] = (
            tensor_power / factorial
        )

    return signature


def test_compiled_solver_dimensions():
    spec, compiled, _ = make_compiled_solver(
        dim=2,
        depth=3,
    )

    assert compiled.D == spec.total_dim
    assert compiled.N == spec.max_level
    assert compiled.max_d == 3
    assert compiled.empty_idx == spec.word_to_index(())


def test_compiled_identity():
    spec, compiled, _ = make_compiled_solver(
        dim=2,
        depth=3,
    )

    expected = TensorElement.eye(spec)

    assert compiled.Id.spec == spec

    np.testing.assert_array_equal(
        compiled.Id._data,
        expected._data,
    )


def test_empty_rule_arrays_have_consistent_shapes():
    spec, compiled, _ = make_compiled_solver(
        dim=2,
        depth=3,
    )

    rows, columns, path_indices, signs = (
        compiled._empty_rule
    )

    assert rows.ndim == 1
    assert columns.shape == rows.shape
    assert path_indices.shape == rows.shape
    assert signs.shape == rows.shape

    assert np.all(rows == compiled.empty_idx)
    assert np.all(columns >= 0)
    assert np.all(columns < spec.total_dim)
    assert np.all(path_indices >= 0)
    assert np.all(path_indices < spec.total_dim)
    assert np.all(np.isin(signs, [-1.0, 1.0]))


def test_linear_part_arrays_have_consistent_shapes():
    spec, compiled, _ = make_compiled_solver(
        dim=2,
        depth=3,
    )

    output_indices, columns, path_indices, coefficients = (
        compiled._lin_part
    )

    assert output_indices.ndim == 1
    assert columns.shape == output_indices.shape
    assert path_indices.shape == output_indices.shape
    assert coefficients.shape == output_indices.shape

    assert np.all(output_indices >= 0)
    assert np.all(output_indices < spec.total_dim)
    assert np.all(columns >= 0)
    assert np.all(columns < spec.total_dim)
    assert np.all(path_indices >= 0)
    assert np.all(path_indices < spec.total_dim)
    assert np.all(np.isfinite(coefficients))


def test_empty_input_returns_single_identity():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=2,
    )

    G = solver.compute([])

    assert len(G) == 1
    assert len(G[0]) == 1
    assert G[0][0].spec == spec

    np.testing.assert_array_equal(
        G[0][0]._data,
        TensorElement.eye(spec)._data,
    )


@pytest.mark.parametrize(
    ("dim", "depth", "n_blocks"),
    [
        (1, 1, 1),
        (1, 3, 4),
        (2, 2, 3),
        (2, 3, 4),
    ],
)
def test_output_table_structure(dim, depth, n_blocks):
    spec, _, solver = make_compiled_solver(dim, depth)

    increments = [
        zero_signature(spec)
        for _ in range(n_blocks)
    ]

    G = solver.compute(increments)

    assert len(G) == n_blocks + 1
    assert all(len(row) == n_blocks + 1 for row in G)

    for i in range(n_blocks + 1):
        for j in range(n_blocks + 1):
            if i <= j:
                assert isinstance(G[i][j], TensorElement)
                assert G[i][j].spec == spec
            else:
                assert G[i][j] is None


def test_diagonal_contains_identity():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=3,
    )

    increments = [
        straight_line_signature(spec, [0.1, -0.2]),
        straight_line_signature(spec, [0.3, 0.1]),
        straight_line_signature(spec, [-0.2, 0.4]),
    ]

    G = solver.compute(increments)
    identity = TensorElement.eye(spec)._data

    for i in range(len(increments) + 1):
        np.testing.assert_array_equal(
            G[i][i]._data,
            identity,
        )


@pytest.mark.parametrize(
    ("dim", "depth", "n_blocks"),
    [
        (1, 1, 4),
        (1, 3, 4),
        (2, 2, 3),
        (2, 3, 3),
    ],
)
def test_zero_path_produces_identity_everywhere(
    dim,
    depth,
    n_blocks,
):
    spec, _, solver = make_compiled_solver(dim, depth)

    increments = [
        zero_signature(spec)
        for _ in range(n_blocks)
    ]

    G = solver.compute(increments)
    identity = TensorElement.eye(spec)._data

    for i in range(n_blocks + 1):
        for j in range(i, n_blocks + 1):
            np.testing.assert_allclose(
                G[i][j]._data,
                identity,
                rtol=1e-13,
                atol=1e-13,
            )


def test_one_block_agrees_with_direct_linear_solve():
    spec, compiled, solver = make_compiled_solver(
        dim=2,
        depth=2,
    )

    increment = straight_line_signature(
        spec,
        displacement=[0.1, -0.2],
    )

    G = solver.compute([increment])

    D = compiled.D
    empty_idx = compiled.empty_idx
    identity = compiled.Id._data

    output_indices, columns, path_indices, coefficients = (
        compiled._lin_part
    )

    A = np.zeros((D, D), dtype=float)

    weights = (
        coefficients
        * increment._data[path_indices]
    )

    np.add.at(
        A,
        (output_indices, columns),
        weights,
    )

    matrix = np.eye(D) - A

    empty_rows, empty_columns, empty_path_indices, empty_signs = (
        compiled._empty_rule
    )

    empty_values = (
        empty_signs
        * increment._data[empty_path_indices]
    )

    matrix[empty_idx, :] = 0.0
    matrix[empty_idx, empty_idx] = 1.0

    np.add.at(
        matrix,
        (empty_rows, empty_columns),
        -empty_values,
    )

    rhs = np.zeros(D, dtype=float)

    weights = compiled.poly.precompute_inc_weights(
        increment
    )

    compiled.poly.eval2_weighted_add_into_data(
        rhs,
        identity,
        identity,
        weights,
    )

    rhs[empty_idx] = identity[empty_idx]

    expected = np.linalg.solve(matrix, rhs)

    np.testing.assert_allclose(
        G[0][1]._data,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_is_repeatable():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=2,
    )

    increments = [
        straight_line_signature(spec, [0.1, -0.2]),
        straight_line_signature(spec, [-0.3, 0.1]),
        straight_line_signature(spec, [0.2, 0.2]),
    ]

    first = solver.compute(increments)
    second = solver.compute(increments)

    for i in range(len(increments) + 1):
        for j in range(i, len(increments) + 1):
            np.testing.assert_array_equal(
                first[i][j]._data,
                second[i][j]._data,
            )


def test_compute_does_not_modify_increments():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=2,
    )

    increments = [
        straight_line_signature(spec, [0.1, -0.2]),
        straight_line_signature(spec, [-0.3, 0.1]),
    ]

    original = [
        increment._data.copy()
        for increment in increments
    ]

    solver.compute(increments)

    for increment, expected in zip(increments, original):
        np.testing.assert_array_equal(
            increment._data,
            expected,
        )


def test_all_computed_coordinates_are_finite():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=3,
    )

    increments = [
        straight_line_signature(spec, [0.05, -0.03]),
        straight_line_signature(spec, [0.02, 0.04]),
        straight_line_signature(spec, [-0.01, 0.02]),
    ]

    G = solver.compute(increments)

    for i in range(len(increments) + 1):
        for j in range(i, len(increments) + 1):
            assert np.all(np.isfinite(G[i][j]._data))


def test_rejects_increment_with_incompatible_spec():
    spec, _, solver = make_compiled_solver(
        dim=2,
        depth=2,
    )

    # This has the same flat dimension:
    # T^2(R^2) has dimension 7;
    # T^1(R^6) also has dimension 7.
    # A size-only check would therefore not catch the mismatch.
    wrong_spec = TensorAlgebraSpec(
        dim=6,
        max_level=1,
    )

    assert wrong_spec.total_dim == spec.total_dim

    incompatible_increment = TensorElement.eye(wrong_spec)

    with pytest.raises(
        ValueError,
        match="spec",
    ):
        solver.compute([incompatible_increment])