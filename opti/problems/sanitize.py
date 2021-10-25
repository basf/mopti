import warnings
from copy import deepcopy
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from opti.constraint import LinearEquality, LinearInequality, NChooseK
from opti.parameter import Parameters
from opti.problem import Problem


def _normalize_parameters_data(
    data: pd.DataFrame, params: Parameters
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Transform the inputs from the domain bounds to the unit range"""

    lo = params.bounds.loc["min"]
    hi = params.bounds.loc["max"]

    lo[lo == -np.inf] = data[lo[lo == -np.inf].index].min(axis=0)
    hi[hi == np.inf] = data[hi[hi == np.inf].index].max(axis=0)

    param_names = [p.name for p in params]

    return (data[param_names] - lo) / (hi - lo), lo.values, (hi - lo).values


class InOrOut(str, Enum):
    IN = "input"
    OUT = "output"


def _sanitize_names(
    n_params: int,
    in_or_out: InOrOut,
) -> Iterable[str]:
    return (f"{in_or_out}_{i}" for i in range(n_params))


def _sanitize_params(
    parameters: Parameters,
    in_or_out: InOrOut,
    sanitized_names: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[Sequence]] = None,
):
    """Sets parameter boundaries to 0 and 1 and replaces parameter names"""
    if sanitized_names is None:
        sanitized_names = _sanitize_names(len(parameters), in_or_out)

    sanitized = []
    for i, p in enumerate(parameters):
        name = f"{in_or_out}_{i}"
        low = -np.inf if np.isneginf(p.low) else 0
        high = np.inf if np.isinf(p.high) else 1
        sanitized.append(type(p)(name, domain=[low, high]))
    return Parameters(sanitized)


def sanitize_problem(problem: Problem, name_of_sanitized: Optional[str] = None):
    """
    This creates a transformation of the problem with sanitized data. Thereby, we try
    to preserve relationships between inputs, outputs, and objectives.

    More precisely, the resulting problem has the following properties:
    - Inputs are named `input_0`, `input_1`, .... Outputs are named analogously.
    - The data is scaled per feature to `[0, 1]`.
    - Coefficients of linear constraints are adapted to the data scaling.
    - Models and evaluatons in terms of `f` are dropped if there are any.

    Currently unsuported are problems with
    - discrete or categorical variables,
    - input constraints that are neither linear nor n-choose-k, or
    - output constraints.

    Args:
        problem: to be sanitized
        name_of_sanitized: name of the resulting problem

    Raises:
        TypeError: in case there are unsupported constraints, data is None, or there are output constraints

    Returns:
        Problem instance with sanitized labels and normalized data
    """
    if problem.data is None:
        raise TypeError("we cannot sanitize a problem without data")
    if problem.output_constraints is not None:
        raise TypeError("output constraints are currently not supported")
    if name_of_sanitized is None:
        name_of_sanitized = "Sanitized"
    if problem.models is not None:
        warnings.warn("models are not sanitized but dropped")

    inputs = _sanitize_params(problem.inputs, InOrOut.IN)
    input_name_map = {pi.name: i.name for pi, i in zip(problem.inputs, inputs)}
    normalized_in_data, xmin, Δx = _normalize_parameters_data(
        problem.data, problem.inputs
    )
    outputs = _sanitize_params(problem.outputs, InOrOut.OUT)
    output_name_map = {pi.name: i.name for pi, i in zip(problem.outputs, outputs)}
    normalized_out_data, ymin, Δy = _normalize_parameters_data(
        problem.data, problem.outputs
    )

    normalized_in_data.columns = inputs.names
    normalized_out_data.columns = outputs.names
    normalized_data = pd.concat([normalized_in_data, normalized_out_data], axis=1)
    normalized_data.reset_index(inplace=True, drop=True)

    objectives = deepcopy(problem.objectives)
    for obj in objectives:
        sanitized_name = output_name_map[obj.name]
        i = outputs.names.index(sanitized_name)
        obj.name = sanitized_name
        obj.parameter = sanitized_name
        obj.target = (obj.target - ymin[i]) / Δy[i]
        if hasattr(obj, "tolerance"):
            obj.tolerance /= Δy[i]

    constraints = deepcopy(problem.constraints)
    if constraints is not None:
        for c in constraints:
            c.names = [input_name_map[n] for n in c.names]
            if isinstance(c, (LinearEquality, LinearInequality)):
                c.lhs = (c.lhs + xmin) * Δx
                if c.rhs > 1e-5:
                    c.lhs = c.lhs / c.rhs
                    c.rhs = 1.0
            elif isinstance(c, NChooseK):
                pass
            else:
                raise TypeError(
                    "sanitizer only supports linear and n-choose-k constraints"
                )

    normalized_problem = Problem(
        name=name_of_sanitized,
        inputs=inputs,
        outputs=outputs,
        objectives=objectives,
        constraints=constraints,
        data=normalized_data,
    )
    return normalized_problem
