from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.generic]

@dataclass(frozen=True)
class TensorAlgebraSpec:
    dim: int
    max_level: int
    dtype: np.dtype = np.float64

    @property
    def total_dim(self) -> int:
        d, N = self.dim, self.max_level
        if d == 1:
            return N + 1
        return (d ** (N + 1) - 1) // (d - 1)

    @property
    def offsets(self) -> Tuple[int, ...]:
        """Determines the first index for each level in the truncated algebra."""
        off = [0]
        for k in range(1, self.max_level + 1):
         off.append(off[-1] + self.dim ** (k - 1))
        return tuple(off)


    def word_to_index(self, word: Tuple[int, ...]) -> int:    
        k = len(word)
        if k == 0:
            return 0
        if k > self.max_level:
            raise ValueError("Word length exceeds max_level")
        
        idx = self.offsets[k]
        d = self.dim
        for j, i in enumerate(word):
            if not (1 <= i <= d):
                raise ValueError("Letters must be in {1,...,dim}")
            idx += (i - 1) * (d ** (k - j - 1))
        return idx


    def index_to_word(self, idx: int) -> Tuple[int, ...]:
        if idx < 0 or idx >= self.total_dim:
            raise IndexError("Index out of range")

        d = self.dim
        for k in range(self.max_level + 1):
            start = self.offsets[k]
            size = d ** k
            if start <= idx < start + size:
                r = idx - start
                letters = []
                for _ in range(k):
                    r, rem = divmod(r, d)
                    letters.append(rem + 1)
                return tuple(reversed(letters))

        raise RuntimeError("Unreachable")

    def iter_words(self):
            """
            Yield all words in T^N(R^d) in graded-lex order:
            (), (1), ..., (d), (1,1), (1,2), ...
            Uses a base-d counter.
            """
            d = self.dim
            N = self.max_level

            # level 0
            yield ()

            # levels 1..N
            for k in range(1, N + 1):
                # base-d counter with digits in {1,...,d}
                digits = [1] * k

                while True:
                    #current word
                    yield tuple(digits)

                    # increment base-d counter
                    i = k - 1
                    while i >= 0 and digits[i] == d:
                        digits[i] = 1
                        i -= 1

                    if i < 0:
                        break

                    digits[i] += 1



class TensorElement:
    __slots__ = ("spec", "_data")

    def __init__(self, spec: TensorAlgebraSpec, data: Optional[Array] = None):
        self.spec = spec
        if data is None:
            self._data = np.zeros(spec.total_dim, dtype=spec.dtype)
        else:
            data = np.asarray(data, dtype=spec.dtype)
            if data.ndim != 1 or data.size != spec.total_dim:
                raise ValueError("Wrong flat array shape")
            self._data = data.copy()

  
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        elif isinstance(key, tuple):
            return self._data[self.spec.word_to_index(key)]
        else:
            raise TypeError("Index must be int or tuple")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._data[key] = value
        elif isinstance(key, tuple):
            self._data[self.spec.word_to_index(key)] = value
        else:
            raise TypeError("Index must be int or tuple")


    def level_view(self, k: int) -> Array:
        if k < 0 or k > self.spec.max_level:
            raise ValueError("Invalid level")

        start = self.spec.offsets[k]
        size = self.spec.dim ** k
        shape = () if k == 0 else (self.spec.dim,) * k
        return self._data[start:start + size].reshape(shape)


    @classmethod
    def zero(cls, spec: TensorAlgebraSpec) -> "TensorElement":
        return cls(spec)

    @classmethod
    def eye(cls, spec: TensorAlgebraSpec) -> "TensorElement":
        x = cls.zero(spec)
        x[()] = 1.0
        return x

    def copy(self) -> "TensorElement":
        return TensorElement(self.spec, self._data)

    def _check_compatible(self, other):
        if self.spec != other.spec:
            raise ValueError("Incompatible tensor algebra specs")

    def __add__(self, other):
        self._check_compatible(other)
        return TensorElement(self.spec, self._data + other._data)

    def __sub__(self, other):
        self._check_compatible(other)
        return TensorElement(self.spec, self._data - other._data)

    def __rmul__(self, scalar):
        return TensorElement(self.spec, scalar * self._data)

    def algebra_product(self, other: "TensorElement") -> "TensorElement":
        self._check_compatible(other)
        spec = self.spec
        out = TensorElement.zero(spec)

        for m in range(spec.max_level + 1):
            Cm = out.level_view(m)
            for i in range(m + 1):
                j = m - i
                Ai = self.level_view(i)
                Bj = other.level_view(j)
                Cm += np.multiply.outer(Ai, Bj)

        return out

    def __matmul__(self, other):
        return self.algebra_product(other)


    def __repr__(self):
        return f"T^{self.spec.max_level}(R^{self.spec.dim}) element"

