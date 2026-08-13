
from tensor_algebra import TensorAlgebraSpec, TensorElement
import numpy as np
import numpy.typing as npt
import esig
from typing import List, Tuple, Optional

def signature_increments_from_path(
    path: np.ndarray,
    M: int, 
    depth: int,
):
    """
    path: array of shape (N+1, d)
    M: block size
    depth: signature depth
    """
    d = path.shape[1]
    spec = TensorAlgebraSpec(dim=d, max_level=depth)

    increments = []

    N = path.shape[0] - 1  # number of increments

    for k in range(0, N, M):
        block = path[k : k + M + 1]

        sig = esig.stream2sig(block, depth)

        inc = TensorElement.zero(spec)
        inc[()] = 1.0

        offset = 1  # skip level-0 (always 1)
        for level in range(1, depth + 1):
            size = d ** level
            inc._data[
                spec.offsets[level] : spec.offsets[level] + size
            ] = sig[offset : offset + size]
            offset += size

        increments.append(inc)

    return spec, increments
