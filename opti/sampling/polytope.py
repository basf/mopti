"""
This module provides functions to uniformly sample points subject to a system of linear
inequality constraints, :math:`Ax <= b` (convex polytope), and linear equality
constraints, :math:`Ax = b` (affine projection).

A comparison of MCMC algorithms to generate uniform samples over a convex polytope is
given in [Chen2018]_. Here, we use the Hit & Run algorithm described in [Smith1984]_.
The R-package `hitandrun`_ provides similar functionality to this module.

References
----------
.. [Chen2018] Chen Y., Dwivedi, R., Wainwright, M., Yu B. (2018) Fast MCMC Sampling
    Algorithms on Polytopes. JMLR, 19(55):1−86
    https://arxiv.org/abs/1710.08165
.. [Smith1984] Smith, R. (1984). Efficient Monte Carlo Procedures for Generating
    Points Uniformly Distributed Over Bounded Regions. Operations Research,
    32(6), 1296-1308.
    www.jstor.org/stable/170949
.. _`hitandrun`: https://cran.r-project.org/web/packages/hitandrun/index.html
"""
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from opti.constraint import Constraints, LinearEquality, LinearInequality
from opti.parameter import Parameters


def _chebyshev_center(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the center of the convex polytope Ax <= b."""
    res = scipy.optimize.linprog(
        np.r_[np.zeros(A.shape[1]), -1],
        A_ub=np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)]),
        b_ub=b,
        bounds=(None, None),
    )
    if not res.success:
        raise Exception("Unable to find Chebyshev center")
    return res.x[:-1]


def _affine_subspace(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a orthonormal basis of the nullspace of A, and a solution to Ax = b.
    This allows to to construct arbitrary solutions as the sum of any vector in the
    nullspace, plus the particular solution.
    """
    N = scipy.linalg.null_space(A)
    xp = np.linalg.pinv(A) @ b
    return N, xp


def _hitandrun(
    A: np.ndarray, b: np.ndarray, x0: np.ndarray
) -> Generator[np.ndarray, None, None]:
    """Generator for uniform sampling from the convex polytope Ax <= b using the
    Hit & Run algorithm described in [Smith1984].
    """
    assert A.shape[1] == len(x0)

    x = x0

    while True:
        # sample random direction from unit hypersphere
        direction = np.random.randn(A.shape[1])
        direction /= np.linalg.norm(direction)

        # distances to each face from the current point in the sampled direction
        D = (b - x @ A.T) / (direction @ A.T)

        # distance to the closest face in and opposite to direction
        lo = np.max(D[D < 0])
        hi = np.min(D[D > 0])

        # make random step
        x += np.random.uniform(lo, hi) * direction
        yield x


def _get_AbNx(parameters, constraints):
    """Get a matrix representation of a system of linear constraints and box bounds.

    Convert all linear constraints and box bounds to matrix form:
        A_eq x = b_eq
        A_ineq x <= b_ineq
    compute the nullspace N and a particular solution xp for A_eq x = b_eq and
    transform the linear constraints to the subspace given by N, so that
        At xt <= bt
        x = xt N.T + xp
    """
    nD = len(parameters)
    nC = len(constraints)

    # matrix form A, b of all linear constraints, A[is_eq] are the linear equality constraints
    A = np.empty((nC, nD))
    b = np.empty(nC)
    is_eq = np.zeros(nC, dtype=bool)

    for i, c in enumerate(constraints):
        lhs = np.zeros(len(parameters))
        for ni, li in zip(c.names, c.lhs):
            lhs[parameters.names.index(ni)] = li
        A[i] = lhs
        b[i] = c.rhs
        is_eq[i] = c.is_equality

    # compute null space and particular solution corresponding to the linear equalities
    if sum(is_eq) > 0:
        N, xp = _affine_subspace(A[is_eq], b[is_eq])
    else:
        N = np.eye(nD)
        xp = np.zeros(nD)

    # add the inequality constraints corresponding to the box bounds
    lower = parameters.bounds.loc["min"]
    upper = parameters.bounds.loc["max"]
    A_ineq = np.row_stack([-np.eye(nD), np.eye(nD)])
    b_ineq = np.r_[-np.array(lower), np.array(upper)]

    if sum(~is_eq) > 0:
        A_ineq = np.r_[A_ineq, A[~is_eq]]
        b_ineq = np.r_[b_ineq, b[~is_eq]]

    # project to the affine subspace given by the equality constraints
    At = A_ineq @ N
    bt = b_ineq - A_ineq @ xp

    return At, bt, N, xp


def polytope_sampling(
    n_samples: int, parameters: Parameters, constraints: Constraints, thin: int = 100
) -> pd.DataFrame:
    """Hit-and-run method to sample uniformly under linear constraints.

    Args:
        n_samples (int): Number of samples.
        parameters (opti.Parameters): Parameter space.
        constraints (opti.Constraints): Constraints on the parameters.
        thin (int, optional): Thinning factor of the generated samples.

    Returns:
        array, shape=(n_samples, dimension): Randomly sampled points.
    """
    for c in constraints:
        if not isinstance(c, (LinearEquality, LinearInequality)):
            raise Exception("Polytope sampling only works for linear constraints.")

    At, bt, N, xp = _get_AbNx(parameters, constraints)

    # hit & run sampling
    x0 = _chebyshev_center(At, bt)
    sampler = _hitandrun(At, bt, x0)
    X = np.empty((n_samples, At.shape[1]))
    for i in range(n_samples):
        for _ in range(thin - 1):
            next(sampler)
        X[i] = next(sampler)

    # project back
    X = X @ N.T + xp
    return pd.DataFrame(columns=parameters.names, data=X)
