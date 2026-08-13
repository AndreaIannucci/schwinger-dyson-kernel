import math

import numpy as np
import pytest

from src.sdker.signatures import signature_increments_from_path
from src.sdker.tensor_algebra import TensorElement


def test_one_dimensional_straight_line_signature():
    path = np.array(
        [
            [0.0],
            [2.0],
        ]
    )

    spec, increments = signature_increments_from_path(
        path=path,
        M=1,
        depth=4,
    )

    assert spec.dim == 1
    assert spec.max_level == 4
    assert len(increments) == 1

    signature = increments[0]

    assert signature[()] == 1.0

    for level in range(1, 5):
        expected = 2.0**level / math.factorial(level)

        assert signature[(1,) * level] == pytest.approx(
            expected
        )


def test_multidimensional_straight_line_signature():
    displacement = np.array([2.0, -3.0])

    path = np.array(
        [
            [0.0, 0.0],
            displacement,
        ]
    )

    spec, increments = signature_increments_from_path(
        path=path,
        M=1,
        depth=3,
    )

    signature = increments[0]

    expected_level_one = displacement
    expected_level_two = (
        np.multiply.outer(displacement, displacement) / 2.0
    )
    expected_level_three = (
        np.einsum(
            "i,j,k->ijk",
            displacement,
            displacement,
            displacement,
        )
        / 6.0
    )

    np.testing.assert_allclose(
        signature.level_view(1),
        expected_level_one,
        rtol=1e-13,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        signature.level_view(2),
        expected_level_two,
        rtol=1e-13,
        atol=1e-13,
    )

    np.testing.assert_allclose(
        signature.level_view(3),
        expected_level_three,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("n_increments", "block_size", "expected_blocks"),
    [
        (1, 1, 1),
        (5, 1, 5),
        (5, 2, 3),
        (6, 2, 3),
        (7, 3, 3),
        (10, 3, 4),
        (10, 20, 1),
    ],
)
def test_number_of_signature_blocks(
    n_increments,
    block_size,
    expected_blocks,
):
    path = np.arange(
        n_increments + 1,
        dtype=float,
    ).reshape(-1, 1)

    _, increments = signature_increments_from_path(
        path=path,
        M=block_size,
        depth=2,
    )

    assert len(increments) == expected_blocks


def test_level_one_records_each_block_displacement():
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [4.0, 3.0],
            [3.0, 5.0],
            [7.0, 4.0],
            [8.0, 6.0],
            [9.0, 3.0],
        ]
    )

    block_size = 3

    _, increments = signature_increments_from_path(
        path=path,
        M=block_size,
        depth=3,
    )

    n_increments = path.shape[0] - 1
    block_starts = range(
        0,
        n_increments,
        block_size,
    )

    for signature, start in zip(increments, block_starts):
        end = min(start + block_size, n_increments)

        expected_displacement = path[end] - path[start]

        np.testing.assert_allclose(
            signature.level_view(1),
            expected_displacement,
            rtol=1e-13,
            atol=1e-13,
        )


def test_final_partial_block_is_included():
    path = np.arange(8, dtype=float).reshape(-1, 1)

    _, increments = signature_increments_from_path(
        path=path,
        M=3,
        depth=2,
    )

    # The blocks contain increments:
    # [0,3], [3,6], and the final partial block [6,7].
    assert len(increments) == 3

    assert increments[0][(1,)] == pytest.approx(3.0)
    assert increments[1][(1,)] == pytest.approx(3.0)
    assert increments[2][(1,)] == pytest.approx(1.0)


def test_every_signature_has_unit_empty_coordinate():
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [-1.0, 3.0],
            [2.0, 4.0],
        ]
    )

    _, increments = signature_increments_from_path(
        path=path,
        M=1,
        depth=3,
    )

    assert all(
        increment[()] == pytest.approx(1.0)
        for increment in increments
    )


def test_returned_elements_use_returned_spec():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
        ]
    )

    spec, increments = signature_increments_from_path(
        path=path,
        M=1,
        depth=2,
    )

    assert spec.dim == 3
    assert spec.max_level == 2

    for increment in increments:
        assert increment.spec == spec


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_chen_identity_across_blocks(block_size):
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [-0.5, 1.0],
            [2.0, -1.0],
            [1.5, 0.5],
        ]
    )

    depth = 3
    n_increments = path.shape[0] - 1

    full_spec, full_increments = (
        signature_increments_from_path(
            path=path,
            M=n_increments,
            depth=depth,
        )
    )

    block_spec, block_increments = (
        signature_increments_from_path(
            path=path,
            M=block_size,
            depth=depth,
        )
    )

    assert full_spec == block_spec
    assert len(full_increments) == 1

    reconstructed = TensorElement.eye(block_spec)

    for increment in block_increments:
        reconstructed = reconstructed @ increment

    np.testing.assert_allclose(
        reconstructed._data,
        full_increments[0]._data,
        rtol=1e-12,
        atol=1e-12,
    )


def test_single_point_path_has_no_increments():
    path = np.array(
        [
            [0.0, 0.0],
        ]
    )

    spec, increments = signature_increments_from_path(
        path=path,
        M=1,
        depth=2,
    )

    assert spec.dim == 2
    assert spec.max_level == 2
    assert increments == []