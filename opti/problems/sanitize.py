import warnings
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from opti.constraint import LinearEquality, LinearInequality, NChooseK
from opti.objective import Objectives
from opti.parameter import Parameters
from opti.problem import Problem


def _normalize_parameter(
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


def _sanitize_params(parameters: Parameters, in_or_out: InOrOut):
    """Sets parameter boundaries to 0 and 1 and replaces parameter names"""
    return Parameters(
        [
            type(p)(name=f"{in_or_out}_{i}", domain=[0, 1])
            for i, p in enumerate(parameters)
        ]
    )


def _sanitize_objective(i: int, o):
    kwargs = dict(
        target=getattr(o, "target", None), exponent=getattr(o, "exponent", None)
    )
    return type(o)(
        parameter=f"output_{i}", **{k: v for k, v in kwargs.items() if v is not None}
    )


def sanitize_problem(problem: Problem, name_of_sanitized: Optional[str] = None):
    """
    The resulting problem has the following properties:
    - Inputs are named `input_0`, `input_1`, ..., and outputs analogously.
    - The data is scaled per feature to [0, 1].
    - Coefficients of linear constraints are adapted to the data scaling.
    - Models are dropped if there are any.

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
    inputs = _sanitize_params(problem.inputs, InOrOut.IN)
    input_name_map = {pi.name: i.name for pi, i in zip(problem.inputs, inputs)}
    outputs = _sanitize_params(problem.outputs, InOrOut.OUT)
    if problem.models is not None:
        warnings.warn("models are not sanitized but dropped")

    normalized_in_data, min_val, normalization_denominator = _normalize_parameter(
        problem.data, problem.inputs
    )
    normalized_out_data, _, _ = _normalize_parameter(problem.data, problem.outputs)
    normalized_in_data.columns = [i.name for i in inputs]
    normalized_out_data.columns = [o.name for o in outputs]
    normalized_data = pd.concat([normalized_in_data, normalized_out_data], axis=1)

    objectives = Objectives(
        [_sanitize_objective(i, o) for i, o in enumerate(problem.objectives)]
    )
    constraints = [] if problem.constraints is None else problem.constraints

    for c in constraints:
        c.names = [input_name_map[n] for n in c.names]
        if isinstance(c, (LinearEquality, LinearInequality)):
            c.lhs = (c.lhs + min_val) * normalization_denominator
            if c.rhs > 1e-5:
                c.lhs = c.lhs / c.rhs
                c.rhs = 1.0
        elif isinstance(c, NChooseK):
            pass
        else:
            raise TypeError(
                "sanitizer only supports only linear and n-choose-k constraints"
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