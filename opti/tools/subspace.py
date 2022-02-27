import numpy as np

from opti.constraint import LinearEquality, LinearInequality
from opti.parameter import Continuous
from opti.problem import Problem
from opti.sampling.polytope import _get_AbNx, vertices


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
    """Project a problem to a linear subspace.

    Linear equality constraints can be removed by projecting the input space to a linear subspace.
    This only works if all constraints are linear.

    Args:
        problem (Problem): _description_

    Returns:
        Problem: an equivalent problem where inputs and constraints are projected to a linear subspace.
    """
    # nothing to do if there is no constraint or no linear equality constraint
    if problem.constraints is None:
        return problem
    if not np.any([isinstance(c, LinearEquality) for c in problem.constraints]):
        return problem

    # method only applicable for all linear constraints
    for c in problem.constraints:
        if not isinstance(c, (LinearEquality, LinearInequality)):
            raise TypeError(
                f"Subspacing only works for linear constraints. Got {type(c)}."
            )

    # determine null space and polytope represention
    At, bt, N, xp = _get_AbNx(problem.inputs, problem.constraints)

    # names of transformed parameters
    names = np.array([f"x{i}" for i in range(N.shape[1])])

    # determine box bounds of transformed parameters from polytope vertices
    V = vertices(At, bt)
    lower = V.min(axis=0)
    upper = V.max(axis=0)

    # the equation system At xt < bt also contains equations for the box bounds
    # (rows with a single 1 or -1) -> remove them as they are not needed.
    keep = (At != 0).sum(axis=1) > 1
    At = At[keep]
    bt = bt[keep]

    # new problem with only inequality constraints
    subproblem = Problem(
        inputs=[
            Continuous(n, domain=[lo, up]) for n, lo, up in zip(names, lower, upper)
        ],
        outputs=problem.outputs,
        objectives=problem.objectives,
        constraints=[
            LinearInequality(names[A != 0], lhs=A[A != 0], rhs=b)
            for A, b in zip(At, bt)
        ],
    )

    return subproblem
