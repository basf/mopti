import warnings
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

from opti.constraint import LinearEquality, LinearInequality, NChooseK
from opti.parameter import Continuous, Parameters
from opti.problem import Problem


def _normalize_parameters_data(
    data: pd.DataFrame, params: Parameters
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Transform data from the domain bounds to the unit range"""

    lo = params.bounds.loc["min"]
    hi = params.bounds.loc["max"]

    lo[lo == -np.inf] = data[lo[lo == -np.inf].index].min(axis=0)
    hi[hi == np.inf] = data[hi[hi == np.inf].index].max(axis=0)

    data = (data[params.names] - lo) / (hi - lo)
    min_vals = lo.values
    delta_vals = (hi - lo).values

    return data, min_vals, delta_vals


def _sanitize_params(parameters: Parameters, prefix: str):
    """Sets parameter boundaries to 0 and 1 and replaces parameter names"""
    sanitized = []
    for i, p in enumerate(parameters):
        name = f"{prefix}_{i}"
        if isinstance(p, Continuous):
            low = -np.inf if np.isneginf(p.low) else 0
            high = np.inf if np.isinf(p.high) else 1
            domain = [low, high]
        else:
            raise TypeError("currently only continuous parameters are supported")
        sanitized.append(type(p)(name, domain))
    return Parameters(sanitized)


def sanitize_problem(problem: Problem) -> Problem:
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
    - nonlinear (in)equality input constraints, or
    - output constraints.

    Args:
        problem: to be sanitized

    Raises:
        TypeError: in case there are unsupported constraints, data is None, or there are output constraints

    Returns:
        Problem instance with sanitized labels and normalized data
    """
    if problem.data is None:
        raise TypeError("we cannot sanitize a problem without data")
    if problem.output_constraints is not None:
        raise TypeError("output constraints are currently not supported")
    if getattr(problem, "f", None) is not None:
        warnings.warn("f is not sanitized but dropped")
    if problem.models is not None:
        warnings.warn("models are not sanitized but dropped")

    inputs = _sanitize_params(problem.inputs, "input")
    input_name_map = {pi.name: i.name for pi, i in zip(problem.inputs, inputs)}
    normalized_in_data, xmin, Δx = _normalize_parameters_data(
        problem.data, problem.inputs
    )
    outputs = _sanitize_params(problem.outputs, "output")
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
        inputs=inputs,
        outputs=outputs,
        objectives=objectives,
        constraints=constraints,
        data=normalized_data,
    )
    return normalized_problem
