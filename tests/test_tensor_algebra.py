import numpy as np
import pytest

from src.sdker.tensor_algebra import TensorAlgebraSpec, TensorElement


@pytest.mark.parametrize(
    ("dim", "max_level", "expected_total_dim", "expected_offsets"),
    [
        (1, 0, 1, (0,)),
        (1, 3, 4, (0, 1, 2, 3)),
        (2, 2, 7, (0, 1, 3)),
        (2, 3, 15, (0, 1, 3, 7)),
        (3, 2, 13, (0, 1, 4)),
    ],
)
def test_spec_dimensions_and_offsets(
    dim,
    max_level,
    expected_total_dim,
    expected_offsets,
):
    spec = TensorAlgebraSpec(dim=dim, max_level=max_level)

    assert spec.total_dim == expected_total_dim
    assert spec.offsets == expected_offsets


@pytest.mark.parametrize(
    ("word", "expected_index"),
    [
        ((), 0),
        ((1,), 1),
        ((2,), 2),
        ((1, 1), 3),
        ((1, 2), 4),
        ((2, 1), 5),
        ((2, 2), 6),
    ],
)
def test_word_to_index(word, expected_index):
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    assert spec.word_to_index(word) == expected_index


def test_word_index_round_trip():
    spec = TensorAlgebraSpec(dim=3, max_level=3)

    words = list(spec.iter_words())

    assert len(words) == spec.total_dim
    assert len(set(words)) == spec.total_dim

    for expected_index, word in enumerate(words):
        index = spec.word_to_index(word)

        assert index == expected_index
        assert spec.index_to_word(index) == word


@pytest.mark.parametrize(
    "word",
    [
        (0,),
        (3,),
        (1, 0),
        (1, 3),
    ],
)
def test_word_to_index_rejects_invalid_letters(word):
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    with pytest.raises(ValueError):
        spec.word_to_index(word)


def test_word_to_index_rejects_excessive_word_length():
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    with pytest.raises(ValueError):
        spec.word_to_index((1, 1, 1))


@pytest.mark.parametrize("index", [-1, 7, 8])
def test_index_to_word_rejects_invalid_index(index):
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    with pytest.raises(IndexError):
        spec.index_to_word(index)


def test_zero_element():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    zero = TensorElement.zero(spec)

    for index in range(spec.total_dim):
        assert zero[index] == 0.0


def test_identity_element():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    identity = TensorElement.eye(spec)

    assert identity[()] == 1.0

    for index in range(1, spec.total_dim):
        assert identity[index] == 0.0


def test_getitem_and_setitem_by_word_and_index():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    element = TensorElement.zero(spec)

    element[(1, 2)] = 3.5

    index = spec.word_to_index((1, 2))

    assert element[(1, 2)] == 3.5
    assert element[index] == 3.5

    element[index] = -2.0

    assert element[(1, 2)] == -2.0


def test_level_view_has_correct_shape_and_values():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    element = TensorElement(
        spec,
        np.arange(spec.total_dim, dtype=float),
    )

    assert element.level_view(0).shape == ()
    assert element.level_view(1).shape == (2,)
    assert element.level_view(2).shape == (2, 2)

    np.testing.assert_array_equal(
        element.level_view(1),
        np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(
        element.level_view(2),
        np.array(
            [
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
    )


@pytest.mark.parametrize("level", [-1, 3])
def test_level_view_rejects_invalid_level(level):
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    element = TensorElement.zero(spec)

    with pytest.raises(ValueError):
        element.level_view(level)


def test_copy_is_independent():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    original = TensorElement.eye(spec)
    copied = original.copy()

    copied[(1,)] = 4.0

    assert original[(1,)] == 0.0
    assert copied[(1,)] == 4.0


def test_addition_and_subtraction():
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    x = TensorElement(
        spec,
        np.arange(spec.total_dim, dtype=float),
    )
    y = TensorElement(
        spec,
        np.ones(spec.total_dim),
    )

    np.testing.assert_allclose(
        (x + y)._data,
        x._data + y._data,
    )

    np.testing.assert_allclose(
        (x - y)._data,
        x._data - y._data,
    )


def test_scalar_multiplication():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    x = TensorElement(
        spec,
        np.arange(spec.total_dim, dtype=float),
    )

    result = 3.0 * x

    np.testing.assert_allclose(
        result._data,
        3.0 * x._data,
    )


def test_tensor_product_identity():
    spec = TensorAlgebraSpec(dim=2, max_level=3)

    rng = np.random.default_rng(123)
    x = TensorElement(
        spec,
        rng.normal(size=spec.total_dim),
    )
    identity = TensorElement.eye(spec)

    np.testing.assert_allclose(
        (identity @ x)._data,
        x._data,
    )

    np.testing.assert_allclose(
        (x @ identity)._data,
        x._data,
    )


def test_tensor_product_associativity():
    spec = TensorAlgebraSpec(dim=2, max_level=3)

    rng = np.random.default_rng(456)

    x = TensorElement(
        spec,
        rng.normal(size=spec.total_dim),
    )
    y = TensorElement(
        spec,
        rng.normal(size=spec.total_dim),
    )
    z = TensorElement(
        spec,
        rng.normal(size=spec.total_dim),
    )

    left = (x @ y) @ z
    right = x @ (y @ z)

    np.testing.assert_allclose(
        left._data,
        right._data,
        rtol=1e-13,
        atol=1e-13,
    )


def test_tensor_product_known_example():
    spec = TensorAlgebraSpec(dim=1, max_level=2)

    x = TensorElement.zero(spec)
    y = TensorElement.zero(spec)

    x[()] = 1.0
    x[(1,)] = 2.0
    x[(1, 1)] = 3.0

    y[()] = 4.0
    y[(1,)] = 5.0
    y[(1, 1)] = 6.0

    result = x @ y

    assert result[()] == 4.0
    assert result[(1,)] == 13.0
    assert result[(1, 1)] == 28.0


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x @ y,
    ],
)
def test_operations_reject_incompatible_specs(operation):
    spec1 = TensorAlgebraSpec(dim=2, max_level=2)
    spec2 = TensorAlgebraSpec(dim=3, max_level=2)

    x = TensorElement.zero(spec1)
    y = TensorElement.zero(spec2)

    with pytest.raises(ValueError):
        operation(x, y)


def test_constructor_copies_input_data():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    data = np.arange(spec.total_dim, dtype=float)

    element = TensorElement(spec, data)
    data[0] = 100.0

    assert element[0] == 0.0


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((7, 1)),
        np.zeros(6),
        np.zeros(8),
    ],
)
def test_constructor_rejects_incorrect_data_shape(data):
    spec = TensorAlgebraSpec(dim=2, max_level=2)

    with pytest.raises(ValueError):
        TensorElement(spec, data)

def test_numpy_integer_index():
    spec = TensorAlgebraSpec(dim=2, max_level=2)
    element = TensorElement.zero(spec)

    index = np.int64(3)

    element[index] = 4.5

    assert element[index] == 4.5
    assert element[3] == 4.5