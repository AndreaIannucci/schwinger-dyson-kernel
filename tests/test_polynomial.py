import numpy as np
import pytest

from src.sdker.combinatorics import get_pairing_cache
from src.sdker.polynomial import (
    compile_poly_numpy,
    compiled_compute_vals_terms,
)
from src.sdker.tensor_algebra import TensorAlgebraSpec, TensorElement


def test_compute_vals_for_empty_word():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    terms = compiled_compute_vals_terms(
        coord=(),
        d_locked=0,
        gub_spec=spec,
        pairing_cache=pairing_cache,
    )

    assert terms == (
        (spec.word_to_index(()), 1.0),
    )


def test_compute_vals_for_one_letter():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    forward_terms = compiled_compute_vals_terms(
        coord=(1,),
        d_locked=0,
        gub_spec=spec,
        pairing_cache=pairing_cache,
        backward=False,
    )

    backward_terms = compiled_compute_vals_terms(
        coord=(1,),
        d_locked=0,
        gub_spec=spec,
        pairing_cache=pairing_cache,
        backward=True,
    )

    index = spec.word_to_index((1,))

    assert forward_terms == ((index, 1.0),)
    assert backward_terms == ((index, -1.0),)


def test_compute_vals_applies_pair_contraction():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    terms = compiled_compute_vals_terms(
        coord=(1, 1),
        d_locked=0,
        gub_spec=spec,
        pairing_cache=pairing_cache,
    )

    assert terms == (
        (spec.word_to_index((1, 1)), 1.0),
        (spec.word_to_index(()), -1.0),
    )


def test_compute_vals_respects_kronecker_constraint():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    terms = compiled_compute_vals_terms(
        coord=(1, 2),
        d_locked=0,
        gub_spec=spec,
        pairing_cache=pairing_cache,
    )

    # The contraction of positions 1 and 2 is excluded because
    # their coordinate labels are different.
    assert terms == (
        (spec.word_to_index((1, 2)), 1.0),
    )


def test_compute_vals_respects_locked_positions():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    terms = compiled_compute_vals_terms(
        coord=(1, 1),
        d_locked=2,
        gub_spec=spec,
        pairing_cache=pairing_cache,
    )

    # Pair (1,2) is forbidden when both positions are locked.
    assert terms == (
        (spec.word_to_index((1, 1)), 1.0),
    )


def test_compute_vals_is_cached():
    spec = TensorAlgebraSpec(dim=1, max_level=2)
    pairing_cache = get_pairing_cache(N=2, max_d=2)

    first = compiled_compute_vals_terms(
        (1, 1),
        0,
        spec,
        pairing_cache,
    )
    second = compiled_compute_vals_terms(
        (1, 1),
        0,
        spec,
        pairing_cache,
    )

    assert first is second


def test_depth_one_polynomial_has_expected_term():
    solution_spec = TensorAlgebraSpec(dim=1, max_level=1)
    tail_spec = TensorAlgebraSpec(dim=1, max_level=0)

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=1,
        max_d=1,
    )

    np.testing.assert_array_equal(poly.out_idx, [1])
    np.testing.assert_array_equal(poly.g1_idx, [0])
    np.testing.assert_array_equal(poly.g2_idx, [0])
    np.testing.assert_array_equal(poly.p_idx, [1])
    np.testing.assert_array_equal(poly.coeff, [-1.0])


def test_compiled_arrays_have_consistent_shapes():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=3,
        max_d=3,
    )

    expected_shape = poly.coeff.shape

    assert poly.out_idx.shape == expected_shape
    assert poly.g1_idx.shape == expected_shape
    assert poly.g2_idx.shape == expected_shape
    assert poly.p_idx.shape == expected_shape

    assert poly.out_idx.ndim == 1
    assert poly.coeff.size > 0


def test_compiled_array_dtypes():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=2,
        max_d=2,
    )

    assert poly.out_idx.dtype == np.int64
    assert poly.g1_idx.dtype == np.int64
    assert poly.g2_idx.dtype == np.int64
    assert poly.p_idx.dtype == np.int64
    assert poly.coeff.dtype == np.float64


def test_compiled_indices_are_in_range():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=3,
        max_d=3,
    )

    dimension = solution_spec.total_dim

    for indices in (
        poly.out_idx,
        poly.g1_idx,
        poly.g2_idx,
        poly.p_idx,
    ):
        assert np.all(indices >= 0)
        assert np.all(indices < dimension)


def test_polynomial_has_no_empty_word_output_terms():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=3)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=2)

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=3,
        max_d=3,
    )

    empty_index = solution_spec.word_to_index(())

    assert not np.any(poly.out_idx == empty_index)


def direct_polynomial_evaluation(poly, gub1, gub2, path):
    out = np.zeros(poly.spec_out.total_dim, dtype=float)

    for (
        out_index,
        g1_index,
        g2_index,
        path_index,
        coefficient,
    ) in zip(
        poly.out_idx,
        poly.g1_idx,
        poly.g2_idx,
        poly.p_idx,
        poly.coeff,
    ):
        out[out_index] += (
            coefficient
            * gub1[g1_index]
            * gub2[g2_index]
            * path[path_index]
        )

    return out


@pytest.mark.parametrize(
    ("dim", "depth"),
    [
        (1, 1),
        (1, 3),
        (2, 2),
        (2, 3),
    ],
)
def test_compiled_evaluation_agrees_with_direct_sum(dim, depth):
    solution_spec = TensorAlgebraSpec(
        dim=dim,
        max_level=depth,
    )
    tail_spec = TensorAlgebraSpec(
        dim=dim,
        max_level=depth - 1,
    )

    poly = compile_poly_numpy(
        gub_spec=solution_spec,
        path_spec=tail_spec,
        N=depth,
        max_d=depth,
    )

    rng = np.random.default_rng(100 * dim + depth)

    gub1 = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )
    gub2 = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )
    path = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )

    expected = direct_polynomial_evaluation(
        poly,
        gub1,
        gub2,
        path,
    )
    actual = poly.eval2(gub1, gub2, path)

    np.testing.assert_allclose(
        actual._data,
        expected,
        rtol=1e-13,
        atol=1e-13,
    )


def test_eval_agrees_with_eval2_using_same_argument():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    poly = compile_poly_numpy(
        solution_spec,
        tail_spec,
        N=2,
        max_d=2,
    )

    rng = np.random.default_rng(123)

    gub = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )
    path = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )

    expected = poly.eval2(gub, gub, path)
    actual = poly.eval(gub, path)

    np.testing.assert_allclose(
        actual._data,
        expected._data,
    )


def test_precomputed_increment_weights():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    poly = compile_poly_numpy(
        solution_spec,
        tail_spec,
        N=2,
        max_d=2,
    )

    path = TensorElement(
        solution_spec,
        np.arange(solution_spec.total_dim, dtype=float),
    )

    weights = poly.precompute_inc_weights(path)

    expected = poly.coeff * path._data[poly.p_idx]

    np.testing.assert_allclose(weights, expected)


def test_weighted_add_accumulates_into_existing_output():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    poly = compile_poly_numpy(
        solution_spec,
        tail_spec,
        N=2,
        max_d=2,
    )

    rng = np.random.default_rng(321)

    gub1 = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )
    gub2 = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )
    path = TensorElement(
        solution_spec,
        rng.normal(size=solution_spec.total_dim),
    )

    initial = rng.normal(size=solution_spec.total_dim)
    actual = initial.copy()

    weights = poly.precompute_inc_weights(path)

    poly.eval2_weighted_add_into_data(
        actual,
        gub1._data,
        gub2._data,
        weights,
    )

    expected = initial + direct_polynomial_evaluation(
        poly,
        gub1,
        gub2,
        path,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-13,
        atol=1e-13,
    )


def test_eval2_rejects_incompatible_solution_spec():
    solution_spec = TensorAlgebraSpec(dim=2, max_level=2)
    tail_spec = TensorAlgebraSpec(dim=2, max_level=1)

    poly = compile_poly_numpy(
        solution_spec,
        tail_spec,
        N=2,
        max_d=2,
    )

    wrong_spec = TensorAlgebraSpec(dim=3, max_level=1)

    correct = TensorElement.eye(solution_spec)
    incorrect = TensorElement.eye(wrong_spec)
    path = TensorElement.eye(solution_spec)

    with pytest.raises(ValueError):
        poly.eval2(incorrect, correct, path)

    with pytest.raises(ValueError):
        poly.eval2(correct, incorrect, path)


def test_depth_zero_produces_empty_polynomial():
    solution_spec = TensorAlgebraSpec(dim=1, max_level=0)
    tail_spec = TensorAlgebraSpec(dim=1, max_level=0)

    poly = compile_poly_numpy(
        solution_spec,
        tail_spec,
        N=0,
        max_d=0,
    )

    assert poly.coeff.size == 0

    identity = TensorElement.eye(solution_spec)
    result = poly.eval(identity, identity)

    np.testing.assert_array_equal(
        result._data,
        np.zeros(solution_spec.total_dim),
    )