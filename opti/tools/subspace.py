import numpy as np

from opti.constraint import LinearEquality, LinearInequality
from opti.parameter import Continuous
from opti.problem import Problem
from opti.sampling.polytope import _get_AbNx


class SubspaceTransform:
    def __init__(self, N, xp):
        """N: nullspace, xp: particular solution."""
        self.N = N
        self.xp = xp

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.N

    def untransform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.N.T + self.xp


def subspace_problem(problem: Problem):
    """Project a problem to a Linear subspaces.

    Linear equality constraints can be removed by projecting the input space to a linear subspace.
    This only works if all constraints are linear.

    Args:
        problem (Problem): _description_

    Returns:
        Problem: a problem where inputs and constraints are projected to a linear subspace.
    """
    if problem.constraints is None:
        return problem
    if not np.any([isinstance(c, LinearEquality) for c in problem.constraints]):
        return problem

    # construct the subspace
    At, bt, N, xp = _get_AbNx(problem.inputs, problem.constraints)
    subspace = SubspaceTransform(N, xp)

    # names of transformed parameters
    names = np.array([f"x{i}" for i in range(N.shape[1])])

    # transformed parameter bounds
    bounds = subspace.transform(problem.inputs.bounds.values).T

    # the equation system At xt < bt also contains equations for the box bounds
    # (rows with a single 1 or -1) -> remove them as they are not needed.
    keep = (At != 0).sum(axis=1) > 1
    At = At[keep]
    bt = bt[keep]

    # new problem with only inequality constraints
    subproblem = Problem(
        inputs=[Continuous(n, b) for n, b in zip(names, bounds)],
        outputs=problem.outputs,
        objectives=problem.objectives,
        constraints=[
            LinearInequality(names[A != 0], lhs=A[A != 0], rhs=b)
            for A, b in zip(At, bt)
        ],
    )

    return subproblem
