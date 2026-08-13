import numpy as np

from sdker.reference_solver import SimpleSolver
from sdker.signatures import signature_increments_from_path
from sdker.solver import make_sdk_solver


def main():
    gamma = np.linspace(
        0.0,
        0.2,
        21,
    ).reshape(-1, 1)

    depth = 3
    block_size = 1

    solution_spec, solver = make_sdk_solver(
        path_dim=gamma.shape[1],
        depth=depth,
    )

    increment_spec, increments = (
        signature_increments_from_path(
            path=gamma,
            M=block_size,
            depth=depth,
        )
    )

    assert increment_spec == solution_spec

    solution = solver.compute(increments)
    sdk_value = solution[0][-1][()]

    reference_value = SimpleSolver(gamma).compute()[0, -1]

    print(f"SDK value:       {sdk_value:.10f}")
    print(f"Reference value: {reference_value:.10f}")
    print(
        "Absolute error:  "
        f"{abs(sdk_value - reference_value):.3e}"
    )


if __name__ == "__main__":
    main()