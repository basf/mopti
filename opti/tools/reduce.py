import json
import os
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from opti import Problem
from opti.constraint import Constraint, Constraints, LinearEquality, LinearInequality
from opti.model import Model, Models
from opti.objective import Objective, Objectives
from opti.parameter import Continuous, Parameter, Parameters

ParametersLike = Union[Parameters, List[Parameter], List[Dict]]
ObjectivesLike = Union[Objectives, List[Objective], List[Dict]]
ConstraintsLike = Union[Constraints, List[Constraint], List[Dict]]
ModelsLike = Union[Models, List[Model], List[Dict]]
DataFrameLike = Union[pd.DataFrame, Dict]
PathLike = Union[str, bytes, os.PathLike]


class ReducedProblem(Problem):
    def __init__(
        self,
        inputs: ParametersLike,
        outputs: ParametersLike,
        objectives: Optional[ObjectivesLike] = None,
        constraints: Optional[ConstraintsLike] = None,
        output_constraints: Optional[ObjectivesLike] = None,
        f: Optional[Callable] = None,
        models: Optional[ModelsLike] = None,
        data: Optional[DataFrameLike] = None,
        optima: Optional[DataFrameLike] = None,
        name: Optional[str] = None,
        _equalities: Optional[List[str]] = None,
        **kwargs,
    ):
        """An optimization problem.

        Args:
            inputs: Input parameters.
            outputs: Output parameters.
            objectives: Optimization objectives. Defaults to minimization.
            constraints: Constraints on the inputs.
            output_constraints: Constraints on the outputs.
            f: Function to evaluate the outputs for given inputs.
                Must have the signature: f(x: pd.DataFrame) -> pd.DataFrame
            data: Experimental data.
            optima: Pareto optima.
            name: Name of the problem.
            _equalities: Only in case of problem reduction due to equality con
                straints. Used to augment the solution of the reduced problem.
        """
        super().__init__(
            inputs,
            outputs,
            objectives,
            constraints,
            output_constraints,
            f,
            models,
            data,
            optima,
            name,
        )

        if isinstance(_equalities, List):
            if len(_equalities) == 0:
                self._equalities = None
            else:
                self._equalities = _equalities
        elif _equalities is None:
            self._equalities = None
        else:
            raise ValueError("_equalities must be a List or None")

        # check if the names and values in self.equalities are valid
        if self.n_equalities > 0:
            for name_lhs, names_rhs, coeffs in self._equalities:
                for name in names_rhs:
                    if name not in self.inputs.names:
                        raise ValueError(
                            f"Equality refers to unknown parameter: {name}"
                        )
                if not all(np.isfinite(coeffs)):
                    raise ValueError("Equality contains non-finite value.")

    @property
    def n_equalities(self) -> int:
        return 0 if self._equalities is None else len(self._equalities)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = super().__str__()
        if self._equalities is not None:
            s = s[:-1] + f"_equalities=\n{self._equalities}\n"
        return "Reduced" + s + ")"

    @staticmethod
    def from_config(config: dict) -> "ReducedProblem":
        """Create a Problem instance from a configuration dict."""
        return ReducedProblem(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        if self._equalities is not None:
            config["_equalities"] = self._equalities
        return config

    @staticmethod
    def from_json(fname: PathLike) -> "ReducedProblem":
        """Read a problem from a JSON file."""
        with open(fname, "rb") as infile:
            config = json.loads(infile.read())
        return ReducedProblem(**config)

    def to_json(self, fname: PathLike) -> None:
        """Save a problem from a JSON file."""
        with open(fname, "wb") as outfile:
            b = json.dumps(self.to_config(), ensure_ascii=False, separators=(",", ":"))
            outfile.write(b.encode("utf-8"))

    @staticmethod
    def augment_data(
        data: pd.DataFrame, _equalities: List[str], names: List[str] = None
    ):
        """Computes augmented DataFrame based on dependencies givern by a set of equalities.

        Args:
            data (DataFrame): data to be augmented.
            equalities (List[str]): Set of equalities used for the augmentation
            names (List[str]): name of all columns given in a certain order to determine the
            order of the columns of the returned data.

        Returns:
            A DataFrame with additional columns (augmented data)
        """
        data = data.copy()

        for name_lhs, names_rhs, coeffs in _equalities:
            data[name_lhs] = coeffs[-1]
            for i, name in enumerate(names_rhs):
                data[name_lhs] += coeffs[i] * data[name]

        if names is not None:
            data = data[names]

        return data


def reduce(problem: Problem) -> ReducedProblem:
    """Reduce a problem with linear equality constraints and linear inequality constraints
    to a subproblem with linear inequality constraints and no linear equality constraints.

    Args:
        problem (Problem): problem to be reduced

    Returns:
        Reduced problem where linear equality constraints have been eliminated

    """
    # check if the problem can be reduced
    if not check_problem_for_reduction(problem):
        return problem

    # find linear equality constraints
    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        problem.constraints
    )

    # only consider continuous inputs
    inputs, otherInputs = find_continuous_inputs(problem.inputs)

    # Assemble Matrix A from equality constraints
    N = len(linearEqualityConstraints)
    M = len(inputs) + 1
    names = np.concatenate((inputs.names, ["rhs"]))

    A_aug = pd.DataFrame(data=np.zeros(shape=(N, M)), columns=names)

    for i in range(len(linearEqualityConstraints)):
        c = linearEqualityConstraints[i]

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

    B[: M - 1, -1] = inputs.bounds.loc["max"].copy()
    B[M - 1 :, -1] = -inputs.bounds.loc["min"].copy()

    # eliminate columns with pivot element
    for i in range(len(pivots)):
        p = pivots[i]
        B[p, :] -= A_aug_rref[i, :]
        B[p + M - 1, :] += A_aug_rref[i, :]

    # build up reduced problem
    _inputs = list(otherInputs.parameters.values())
    for i in range(len(inputs)):
        # add all inputs that were not eliminated
        if i not in pivots:
            _inputs.append(inputs[names[i]])
    _inputs = Parameters(_inputs)

    _constraints = otherConstraints.constraints
    for i in pivots:
        ind = np.where(B[i, :-1] != 0)[0]
        if len(ind) > 0 and B[i, -1] < np.inf:
            c = LinearInequality(names=list(names[ind]), lhs=B[i, ind], rhs=B[i, -1])
            _constraints.append(c)
        else:
            if B[i, -1] < -1e-16:
                raise Warning(
                    "There is no solution in the domain of the variables "
                    "that fulfills the constraints."
                )

        ind = np.where(B[i + M - 1, :-1] != 0)[0]
        if len(ind) > 0 and B[i + M - 1, -1] < np.inf:
            c = LinearInequality(
                names=list(names[ind]), lhs=B[i + M - 1, ind], rhs=B[i + M - 1, -1]
            )
            _constraints.append(c)
        else:
            if B[i + M - 1, -1] < -1e-16:
                raise Warning(
                    "There is no solution in the domain of the variables "
                    "that fulfills the constraints."
                )
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
    # We do not drop the values of the eliminated variables.
    # drop = []
    # if _data is not None:
    #    for col in _data.columns:
    #        if col not in _inputs.names and col not in problem.outputs.names:
    #            drop.append(col)
    #    _data = _data.drop(columns=drop)

    _models = problem.models
    # We ignorie the models attribute for now
    # if _models is not None:
    #    pass

    # Wrap f, if defined
    _f = None
    if "f" in list(vars(problem).keys()):
        if problem.f is not None:

            def _f(X: pd.DataFrame) -> pd.DataFrame:
                return problem.f(
                    ReducedProblem.augment_data(X, _equalities, inputs.names)
                )

    _problem = ReducedProblem(
        inputs=deepcopy(_inputs),
        outputs=deepcopy(problem.outputs),
        objectives=deepcopy(problem.objectives),
        constraints=deepcopy(_constraints),
        f=deepcopy(_f),
        models=deepcopy(_models),
        data=deepcopy(_data),
        optima=deepcopy(problem.optima),
        name=deepcopy(problem.name),
        _equalities=deepcopy(_equalities),
    )

    # remove remaining dependencies of eliminated inputs from the problem
    _problem = remove_eliminated_inputs(_problem)

    return _problem


def find_linear_equality_constraints(constraints: Constraints) -> List[Constraints]:
    """Returns a list two Constraints objects - one containing all linear equality constraints
    the other one containing all the rest.
    """
    linearEqualityConstraints = []
    otherConstraints = []
    for c in constraints:
        if isinstance(c, LinearEquality):
            linearEqualityConstraints.append(c)
        else:
            otherConstraints.append(c)
    return [Constraints(linearEqualityConstraints), Constraints(otherConstraints)]


def find_continuous_inputs(inputs: Parameters) -> List[Parameters]:
    """Returns a list of two Parameters objects - one containing all continuous inputs, "
    the other one containing all the rest.
    """
    contInputs = []
    otherInputs = []
    for p in inputs:
        if isinstance(p, Continuous):
            contInputs.append(p)
        else:
            otherInputs.append(p)
    return [Parameters(contInputs), Parameters(otherInputs)]


def check_problem_for_reduction(problem: Problem) -> bool:
    """Checks if the reduction can be applied or if a trivial case is present."""
    # Are there any constraints?
    if problem.constraints is None:
        return False

    # Are there any linear equality constraints?
    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        problem.constraints
    )

    if len(linearEqualityConstraints) == 0:
        return False

    # identify continuous inputs
    inputs, otherInputs = find_continuous_inputs(problem.inputs)

    if len(inputs) == 0:
        return False

    # check if there are invalid equality constraints (i.e. constraints
    # containing non-continuous parameters)
    for c in linearEqualityConstraints:
        for name in c.names:
            if name not in inputs.names:
                raise RuntimeError(
                    f"Linear equality constraint {c} contains "
                    "a non-continuous or non-existing parameter. Problem reduction for this situation"
                    " is not supported."
                )
    return True


def check_existence_of_solution(A_aug):
    """Given an augmented coefficient matrix this function determines the existence
    (and uniqueness) of solution using the rank theorem."""
    A = A_aug[:, :-1]
    b = A_aug[:, -1]
    len_inputs = np.shape(A)[1]

    # catch special cases
    rk_A_aug = np.linalg.matrix_rank(A_aug)
    rk_A = np.linalg.matrix_rank(A)

    if rk_A == rk_A_aug:
        if rk_A == len_inputs:
            x = np.linalg.solve(A, b)
            raise Warning(
                "There is a unique solution x for the linear equality constraints: x="
                + str(x)
            )
        else:
            pass
    elif rk_A < rk_A_aug:
        raise Warning("There is no solution fulfilling the linear equality constraints")
    else:
        raise Warning(
            "Something went wrong. Rank of coefficient matrix must not be "
            "larger than rank of augmented coefficient matrix"
        )


def remove_eliminated_inputs(problem: ReducedProblem) -> ReducedProblem:
    """Eliminates remaining occurences of eliminated inputs in linear constraints."""
    inputs_names = problem.inputs.names
    M = len(inputs_names)

    # write the equalities for the backtransformation into one matrix
    inputs_dict = {inputs_names[i]: i for i in range(M)}

    # build up dict from problem.equalities e.g. {"xi1": [coeff(xj1), ..., coeff(xjn)], ... "xik":...}
    coeffs_dict = {}
    for i, e in enumerate(problem._equalities):
        coeffs = np.zeros(M + 1)
        for j, name in enumerate(e[1]):
            coeffs[inputs_dict[name]] = e[2][j]
        coeffs[-1] = e[2][-1]
        coeffs_dict[e[0]] = coeffs

    constraints = []
    for c in problem.constraints:
        # Nonlinear (in)equalities not supported
        if not isinstance(c, LinearEquality) and not isinstance(c, LinearInequality):
            warnings.warn(
                "Elimination of variables is only supported for "
                "LinearEquality and LinearInequality constraints. Probably variables have "
                "to be eliminated manually",
                UserWarning,
            )
            constraints.append(c)

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
                raise RuntimeError("Linear constraints cannot be fulfilled.")
            elif np.isinf(_c.rhs):
                pass
            else:
                constraints.append(_c)
    problem.constraints = Constraints(constraints)

    return problem


def rref(A: np.ndarray, tol=1e-8) -> np.ndarray:
    """Computes the reduced row echelon form of a Matrix

    Args:
        A (ndarray): 2d array representing a matrix.
        tol (float): tolerance for rounding to 0

    Returns:
        [A_rref, pivots], where A_rref is the reduced row echelon form of A and pivots
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
    return [np.round(A, prec), pivots]
