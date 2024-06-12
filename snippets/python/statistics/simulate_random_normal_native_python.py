"""

NOTES: This uses the box-muller transform
NOTES: I discovered this through this stack overflow question: "https://stackoverflow.com/questions/70780758/how-to-generate-random-normal-distribution-without-numpy-google-interview"
"""

import math
import random
from typing import Generator


def rnorm(n: int, mean: float = 0.0, sd: float = 1.0) -> Generator[float, None, None]:
    """Simulates `n` independent draws from a univariate Gaussian
    distribution with mean `mean` and standard deviation `sd`
    """
    if not sd > 0:
        raise ValueError("Standard deviation must be positive")
    for _ in range(n):
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        yield sd * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2) + mean


if __name__ == "__main__":
    # compare function output to numpy
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    N_SAMPLES: int = 1_000_000
    MEAN: float = -50
    STD_DEV: float = 10

    start_time = time.perf_counter()
    native_python_rnorm = list(
        rnorm(
            N_SAMPLES,
            mean=MEAN,
            sd=STD_DEV,
        )
    )
    native_finished = time.perf_counter()
    numpy_rnorm = np.random.normal(MEAN, STD_DEV, N_SAMPLES)
    numpy_finished = time.perf_counter()
    print(
        f"""
    n samples generated: {N_SAMPLES:,}
    Native python finished in {native_finished-start_time:,.2f} seconds 
    Numpy finished in {numpy_finished-native_finished:,.2f} seconds
        """
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        native_python_rnorm,
        bins=100,
        color="blue",
        edgecolor="blue",
        alpha=0.5,
        label="Native Python",
    )
    ax.hist(
        numpy_rnorm,
        bins=100,
        color="red",
        edgecolor="red",
        alpha=0.5,
        label="numpy.random.normal",
    )
    ax.legend()
    plt.show()
