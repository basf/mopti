import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd

from opti import Problem
from opti.constraint import Constraints, LinearEquality, LinearInequality
from opti.parameter import Continuous, Parameters


class AffineTransform:
    def __init__(self, equalities):
        self.equalities = equalities

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restore eliminated parameters in a dataframe."""
        data = data.copy()
        for name_lhs, names_rhs, coeffs in self.equalities:
            data[name_lhs] = coeffs[-1]
            for i, name in enumerate(names_rhs):
                data[name_lhs] += coeffs[i] * data[name]
        return data

    def drop_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop eliminated parameters from a DataFrame."""
        drop = []
        for name_lhs, _, _ in self.equalities:
            if name_lhs in data.columns:
                drop.append(name_lhs)
        return data.drop(columns=drop)


def reduce_problem(problem: Problem) -> Tuple[Problem, AffineTransform]:
    """Reduce a problem with linear constraints to a subproblem where linear equality constraints are eliminated.

    Args:
        problem (Problem): problem to be reduced

    Returns:
        (problem, trafo). Problem is the reduced problem where linear equality constraints
        have been eliminated. trafo is the according transformation.
    """
    # check if the problem can be reduced
    if not check_problem_for_reduction(problem):
        return problem, AffineTransform([])

    # find linear equality constraints
    linear_equalities, other_constraints = find_linear_equalities(problem.constraints)

    # only consider continuous inputs
    continuous_inputs, other_inputs = find_continuous_inputs(problem.inputs)

    # assemble Matrix A from equality constraints
    N = len(linear_equalities)
    M = len(continuous_inputs) + 1
    names = np.concatenate((continuous_inputs.names, ["rhs"]))

    A_aug = pd.DataFrame(data=np.zeros(shape=(N, M)), columns=names)

    for i in range(len(linear_equalities)):
        c = linear_equalities[i]

        A_aug.loc[i, c.names] = c.lhs
        A_aug.loc[i, "rhs"] = c.rhs
    A_aug = A_aug.values

    # catch special cases
    check_existence_of_solution(A_aug)

    # bring A_aug to reduced row-echelon form
    A_aug_rref, pivots = rref(A_aug)
    pivots = np.array(pivots)
    A_aug_rref = np.array(A_aug_rref).astype(np.float64)

    # formulate box bounds as linear inequality constraints in matrix form
    B = np.zeros(shape=(2 * (M - 1), M))
    B[: M - 1, : M - 1] = np.eye(M - 1)
    B[M - 1 :, : M - 1] = -np.eye(M - 1)

    B[: M - 1, -1] = continuous_inputs.bounds.loc["max"].copy()
    B[M - 1 :, -1] = -continuous_inputs.bounds.loc["min"].copy()

    # eliminate columns with pivot element
    for i in range(len(pivots)):
        p = pivots[i]
        B[p, :] -= A_aug_rref[i, :]
        B[p + M - 1, :] += A_aug_rref[i, :]

    # build up reduced problem
    _inputs = list(other_inputs.parameters.values())
    for i in range(len(continuous_inputs)):
        # add all inputs that were not eliminated
        if i not in pivots:
            _inputs.append(continuous_inputs[names[i]])
    _inputs = Parameters(_inputs)

    _constraints = other_constraints.constraints
    for i in pivots:
        # reduce equation system of upper bounds
        ind = np.where(B[i, :-1] != 0)[0]
        if len(ind) > 0 and B[i, -1] < np.inf:
            c = LinearInequality(names=list(names[ind]), lhs=B[i, ind], rhs=B[i, -1])
            _constraints.append(c)
        else:
            if B[i, -1] < -1e-16:
                raise Exception("There is no solution that fulfills the constraints.")

        # reduce equation system of lower bounds
        ind = np.where(B[i + M - 1, :-1] != 0)[0]
        if len(ind) > 0 and B[i + M - 1, -1] < np.inf:
            c = LinearInequality(
                names=list(names[ind]), lhs=B[i + M - 1, ind], rhs=B[i + M - 1, -1]
            )
            _constraints.append(c)
        else:
            if B[i + M - 1, -1] < -1e-16:
                raise Exception("There is no solution that fulfills the constraints.")

    _constraints = Constraints(_constraints)

    # assemble equalities
    _equalities = []
    for i in range(len(pivots)):
        name_lhs = names[pivots[i]]
        names_rhs = []
        coeffs = []

        for j in range(len(names) - 1):
            if A_aug_rref[i, j] != 0 and j != pivots[i]:
                coeffs.append(-A_aug_rref[i, j])
                names_rhs.append(names[j])

        coeffs.append(A_aug_rref[i, -1])

        _equalities.append([name_lhs, names_rhs, coeffs])

    _data = problem.data
    trafo = AffineTransform(_equalities)

    _models = problem.models
    if _models is not None:
        warnings.warn("Models are currently not adapted in reduce_problem.")

    if hasattr(problem, "f") and problem.f is not None:

        def _f(X: pd.DataFrame) -> pd.DataFrame:
            return problem.f(trafo.augment_data(X))

    else:
        _f = None

    _problem = Problem(
        inputs=_inputs,
        outputs=deepcopy(problem.outputs),
        objectives=deepcopy(problem.objectives),
        constraints=deepcopy(_constraints),
        f=_f,
        models=_models,
        data=_data,
        optima=deepcopy(problem.optima),
        name=deepcopy(problem.name),
    )

    # remove remaining dependencies of eliminated inputs from the problem
    _problem = remove_eliminated_inputs(_problem, trafo)
    return _problem, trafo


def find_linear_equalities(constraints: Constraints) -> Tuple[Constraints, Constraints]:
    """Separate constraints into linear equalities and all other constraints."""
    linear_equalities = [c for c in constraints if isinstance(c, LinearEquality)]
    other_constraints = [c for c in constraints if not isinstance(c, LinearEquality)]
    return Constraints(linear_equalities), Constraints(other_constraints)


def find_continuous_inputs(inputs: Parameters) -> Tuple[Parameters, Parameters]:
    """Separate parameters into continuous and all other parameters."""
    continous_inputs = [p for p in inputs if isinstance(p, Continuous)]
    other_inputs = [p for p in inputs if not isinstance(p, Continuous)]
    return Parameters(continous_inputs), Parameters(other_inputs)


def check_problem_for_reduction(problem: Problem) -> bool:
    """Check if the reduction can be applied or if a trivial case is present."""
    # are there any constraints?
    if problem.constraints is None:
        return False

    # are there any linear equality constraints?
    linear_equalities, _ = find_linear_equalities(problem.constraints)
    if len(linear_equalities) == 0:
        return False

    # are there continuous inputs
    continuous_inputs, _ = find_continuous_inputs(problem.inputs)
    if len(continuous_inputs) == 0:
        return False

    # check that equality constraints only contain continuous inputs
    for c in linear_equalities:
        for name in c.names:
            if name not in continuous_inputs.names:
                raise Exception(
                    f"Linear equality constraint {c} contains a non-continuous parameter. Problem reduction is not supported."
                )
    return True


def check_existence_of_solution(A_aug):
    """Given an augmented coefficient matrix this function determines the existence (and uniqueness) of solution using the rank theorem."""
    A = A_aug[:, :-1]
    b = A_aug[:, -1]
    len_inputs = np.shape(A)[1]

    # catch special cases
    rk_A_aug = np.linalg.matrix_rank(A_aug)
    rk_A = np.linalg.matrix_rank(A)

    if rk_A == rk_A_aug:
        if rk_A < len_inputs:
            return  # all good
        else:
            x = np.linalg.solve(A, b)
            raise Exception(
                f"There is a unique solution x for the linear equality constraints: x={x}"
            )
    elif rk_A < rk_A_aug:
        raise Exception(
            "There is no solution fulfilling the linear equality constraints."
        )


def remove_eliminated_inputs(problem: Problem, transform: AffineTransform) -> Problem:
    """Eliminates remaining occurences of eliminated inputs in linear constraints."""
    inputs_names = problem.inputs.names
    M = len(inputs_names)

    # write the equalities for the backtransformation into one matrix
    inputs_dict = {inputs_names[i]: i for i in range(M)}

    # build up dict from problem.equalities e.g. {"xi1": [coeff(xj1), ..., coeff(xjn)], ... "xik":...}
    coeffs_dict = {}
    for i, e in enumerate(transform.equalities):
        coeffs = np.zeros(M + 1)
        for j, name in enumerate(e[1]):
            coeffs[inputs_dict[name]] = e[2][j]
        coeffs[-1] = e[2][-1]
        coeffs_dict[e[0]] = coeffs

    constraints = []
    for c in problem.constraints:
        # Nonlinear constraints supported
        if not isinstance(c, (LinearEquality, LinearInequality)):
            raise Exception(
                "Elimination of variables is only supported for LinearEquality and LinearInequality constraints."
            )

        # no changes, if the constraint does not contain eliminated inputs
        elif all(name in inputs_names for name in c.names):
            constraints.append(c)

        # remove inputs from the constraint that were eliminated from the inputs before
        else:
            _names = np.array(inputs_names)
            _rhs = c.rhs

            # create new lhs and rhs from the old one and knowledge from problem._equalities
            _lhs = np.zeros(M)
            for j, name in enumerate(c.names):
                if name in inputs_names:
                    _lhs[inputs_dict[name]] += c.lhs[j]
                else:
                    _lhs += c.lhs[j] * coeffs_dict[name][:-1]
                    _rhs -= c.lhs[j] * coeffs_dict[name][-1]

            _names = _names[np.abs(_lhs) > 1e-16]
            _lhs = _lhs[np.abs(_lhs) > 1e-16]

            # create new Constraints
            if isinstance(c, LinearEquality):
                _c = LinearEquality(_names, _lhs, _rhs)
            else:
                _c = LinearInequality(_names, _lhs, _rhs)

            # check if constraint is always fulfilled/not fulfilled
            if len(_c.names) == 0 and _c.rhs >= 0:
                pass
            elif len(_c.names) == 0 and _c.rhs < 0:
                raise Exception("Linear constraints cannot be fulfilled.")
            elif np.isinf(_c.rhs):
                pass
            else:
                constraints.append(_c)
    problem.constraints = Constraints(constraints)

    return problem


def rref(A: np.ndarray, tol=1e-8) -> Tuple[np.ndarray, List[int]]:
    """Computes the reduced row echelon form of a Matrix

    Args:
        A (ndarray): 2d array representing a matrix.
        tol (float): tolerance for rounding to 0

    Returns:
        (A_rref, pivots), where A_rref is the reduced row echelon form of A and pivots
        is a numpy array containing the pivot columns of A_rref
    """
    A = np.array(A, dtype=np.float64)
    n, m = np.shape(A)

    col = 0
    row = 0
    pivots = []

    for col in range(m):
        # does a pivot element exist?
        if all(np.abs(A[row:, col]) < tol):
            pass
        # if yes: start elimination
        else:
            pivots.append(col)
            max_row = np.argmax(np.abs(A[row:, col])) + row
            # switch to most stable row
            A[[row, max_row], :] = A[[max_row, row], :]
            # normalize row
            A[row, :] /= A[row, col]
            # eliminate other elements from column
            for r in range(n):
                if r != row:
                    A[r, :] -= A[r, col] / A[row, col] * A[row, :]
            row += 1

    prec = int(-np.log10(tol))
    return np.round(A, prec), pivots
