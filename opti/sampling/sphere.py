import numpy as np


def sample(dimension: int, n_samples: int = 1, positive: bool = False) -> np.ndarray:
    """Sample uniformly from the unit hypersphere.

    Args:
        dimension (int): Number of dimensions.
        n_samples (int): Number of samples to draw.
        positive (bool): Sample from the non-negative unit-sphere.

    Returns:
        array, shape=(n_samples, dimesnion): Random samples from the unit simplex.
    """
    x = np.random.normal(0, 1, size=(n_samples, dimension))
    x = x / np.sum(x ** 2, axis=1, keepdims=True) ** 0.5
    if positive:
        x *= -2 * (x < 0) + 1
    return x
