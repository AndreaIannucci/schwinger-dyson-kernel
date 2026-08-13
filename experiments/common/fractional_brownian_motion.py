import numpy as np
import numpy.typing as npt
import esig 
from typing import List, Tuple, Optional

def fbm_davies_harte(
    dim: int,
    n_steps: int,
    H: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate d-dimensional fractional Brownian motion on [0, 1]
    using the Davies–Harte method.

    Parameters
    ----------
    dim : int
        Dimension of the fBm.
    n_steps : int
        Number of steps (n_steps + 1 time points).
    H : float
        Hurst parameter, 0 < H < 1.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    B : (n_steps+1, dim) ndarray
        fBm paths with B[0] = 0.
    """

    if dim <= 0:
            raise ValueError("dim must be positive")
    
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1)")

    if rng is None:
        rng = np.random.default_rng(0)

    

    # --- fGn autocovariance ---
    k = np.arange(n_steps)
    gamma = 0.5 * (
        np.abs(k + 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
    )

    def one_fgn():
        # Circulant embedding
        r = np.zeros(2 * n_steps)
        r[:n_steps] = gamma
        r[n_steps + 1 :] = gamma[1:][::-1]

        lam = np.fft.fft(r).real
        lam = np.maximum(lam, 0.0)

        W = np.zeros(2 * n_steps, dtype=np.complex128)
        W[0] = rng.normal()
        W[n_steps] = rng.normal()

        a = rng.normal(size=n_steps - 1)
        b = rng.normal(size=n_steps - 1)
        W[1:n_steps] = (a + 1j * b) / np.sqrt(2)
        W[n_steps + 1 :] = np.conj(W[1:n_steps][::-1])

        x = np.sqrt(2 * n_steps) * np.fft.ifft(np.sqrt(lam) * W).real
        return x[:n_steps]

    
    # Generate independent components
    dB = np.stack([one_fgn() for _ in range(dim)], axis=1)

    # Scale to [0,1]
    dt = 1.0 / n_steps
    dB *= dt**H

    # Integrate
    B = np.zeros((n_steps + 1, dim))
    B[1:] = np.cumsum(dB, axis=0)

    return  B

