import numpy as np


def is_pareto_efficient(A: np.ndarray) -> np.ndarray:
    """Find the Pareto-efficient points in a set of objective vectors.

    Args:
        A (2D-array, shape=(samples, dimension)): Objective vectors.

    Returns:
        1D-array of bools: Boolean mask for the Pareto efficient points in A.
    """
    efficient = np.ones(len(A), dtype=bool)
    idx = np.arange(len(A))
    for i, a in enumerate(A):
        if not efficient[i]:
            continue
        # set all *other* efficent points to False, if they are not strictly better in at least one objective
        efficient[efficient] = np.any(A[efficient] < a, axis=1) | (i == idx[efficient])
    return efficient


def pareto_front(A: np.ndarray) -> np.ndarray:
    """Find the Pareto-efficient points in a set of objective vectors.

    Args:
        A (2D-array, shape=(samples, dimension)): Objective vectors.

    Returns:
        2D-array: Pareto efficient points in A.
    """
    return A[is_pareto_efficient(A)]


def crowding_distance(A):
    """Crowding distance indicator.

    The crowding distance is defined for each point in a Pareto front as the average
    side length of the cuboid formed by the neighbouring points.

    Reference:
        [Kalyanmoy Deb (2000) A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II](https://link.springer.com/chapter/10.1007/3-540-45356-3_83)

    Args:
        A (2D-array): Set of points representing a Pareto front.
            Pareto-efficiency is assumed but not checked.

    Returns:
        array: Crowding distance indicator for each point in the front.
    """
    A = pareto_front(A)
    N, m = A.shape

    # no crowding distance for 2 points
    if N <= 2:
        return np.full(N, np.inf)

    # sort points along each objective
    sort = np.argsort(A, axis=0)
    A = A[sort, np.arange(m)]

    # normalize all objectives
    norm = np.max(A, axis=0) - np.min(A, axis=0)
    A = A / norm
    A[:, norm == 0] = 0  # handle min = max

    # distance to previous and to next point along each objective
    d = np.diff(A, axis=0)
    inf = np.full((1, m), np.inf)
    d0 = np.concatenate([inf, d])
    d1 = np.concatenate([d, inf])

    # TODO: handle cases with duplicate objective values leading to 0 distances

    # cuboid side length = distance between previous and next point
    unsort = np.argsort(sort, axis=0)
    cuboid = d0[unsort, np.arange(m)] + d1[unsort, np.arange(m)]
    return np.mean(cuboid, axis=1)


def generational_distance(
    A: np.ndarray, R: np.ndarray, p: float = 1, clip: bool = True
) -> float:
    r"""Generational distance indicator.

    The generational distance (GD) is defined as the average distance of points in the
    approximate front A to a reference front R.
    .. math:: \mathrm{GD}(A, R) = (\sum\limits_{a \in A} d(a, R)^p)^{1/p} / N_A
    where d(a, R) is the euclidean distance of point a to the reference front, N_A is
    the number of points in A. GD is a measure of convergence.

    Reference:
        [David Van Veldhuizen+ (2000) Multiobjective Evolutionary Algorithms: Analyzing the State-of-the-Art](http://dx.doi.org/10.1162/106365600568158)

    Args:
        A (2D-array): Set of points representing an approximate Pareto front.
        R (2D-array): Set of points representing a reference Pareto front.
        p (int, optional): Order of the p-norm for averaging over distances.
            Defaults to 1, yielding the standard average.
        clip (bool, optional): Flag for using the modfied generational distance, which
            prevents negative values for points that are non-dominated by the reference
            front.

    Returns:
        float: Generational distance indicator.
    """
    A = pareto_front(A)
    distances = A[:, np.newaxis] - R[np.newaxis]
    if clip:
        distances = distances.clip(0, None)
    distances = np.linalg.norm(distances, axis=2).min(axis=1)
    return np.linalg.norm(distances, p) / len(A)


def inverted_generational_distance(A: np.ndarray, R: np.ndarray, p: float = 1) -> float:
    """Inverted generational distance indicator.

    The inverted generational distance (IGD) is defined as the average distance of
    points in the reference front R to an approximate front A.
    IGD is a measure of convergence, spread and distribution of the approximate front.

    Reference:
        [CA. Coello Coello+ (2004) A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm](https://doi.org/10.1007/978-3-540-24694-7_71)

    Args:
        A (2D-array): Set of points representing an approximate Pareto front.
        R (2D-array): Set of points representing a reference Pareto front.
        p (int, optional): Order of the p-norm for averaging over distances.
            Defaults to 1, yielding the standard average.

    Returns:
        float: Inverted generational distance indicator.
    """
    return generational_distance(R, A, p, clip=False)
