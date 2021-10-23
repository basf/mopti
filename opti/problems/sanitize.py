import warnings
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from opti.constraint import LinearEquality, LinearInequality, NChooseK
from opti.objective import CloseToTarget, Maximize, Minimize, Objectives
from opti.parameter import Parameter, Parameters
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
    if domains is None:
        domains = ((0, 1) for _ in range(len(parameters)))
    return Parameters(
        [
            type(p)(
                name=s_name,
                domain=d,
            )
            for (s_name, d, p) in zip(sanitized_names, domains, parameters)
        ]
    )


def _sanitize_objective(
    i: int,
    problem_obj: Union[Minimize, Maximize, CloseToTarget],
    sanitized_output: Parameter,
    problem_output: Parameter,
    problem_data: pd.DataFrame,
):
    s_lo, s_hi = sanitized_output.domain
    p_lo, p_hi = problem_output.domain
    if np.isneginf(p_lo):
        p_lo = problem_data[problem_output.name].min(axis=0)
    if np.isinf(p_hi):
        p_hi = problem_data[problem_output.name].max(axis=0)
    target = (problem_obj.target - p_lo) / (p_hi - p_lo) * (s_hi - s_lo) + s_lo
    target = min(max(target, s_lo), s_hi)
    tolerance = getattr(problem_obj, "tolerance", None)
    if tolerance is not None:
        tolerance = tolerance / (s_hi - s_lo)

    kwargs = dict(
        target=target,
        exponent=getattr(problem_obj, "exponent", None),
        tolerance=tolerance,
    )
    return type(problem_obj)(
        parameter=f"output_{i}", **{k: v for k, v in kwargs.items() if v is not None}
    )


def sanitize_problem(problem: Problem, name_of_sanitized: Optional[str] = None):
    """
    This creates a transformation of the problem with sanitized data. Thereby, we try
    to preserve relationships between inputs, outputs, and objectives.

    More precisely, the resulting problem has the following properties:
    - Inputs are named `input_0`, `input_1`, .... Outputs are named analogously.
    - The input-data is scaled per feature to `[0, 1]`.
    - If the problem can be evaluated, the output-data are the evaluations of the scaled inputs.
    - If the problem cannot be evaluated, the output-data is scaled to `[0, 1]`.
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

    if problem.models is not None:
        warnings.warn("models are not sanitized but dropped")

    normalized_in_data, min_val, normalization_denominator = _normalize_parameters_data(
        problem.data, problem.inputs
    )
    if not hasattr(problem, "f") or problem.f is None:
        outputs = _sanitize_params(problem.outputs, InOrOut.OUT)
        normalized_out_data, _, _ = _normalize_parameters_data(
            problem.data, problem.outputs
        )
    else:
        names = list(_sanitize_names(problem.n_outputs, InOrOut.OUT))

        normalized_out_data = pd.DataFrame(
            problem.f(normalized_in_data.values),
            index=normalized_in_data.index,
            columns=names,
        )
        domains = zip(normalized_out_data.min(), normalized_out_data.max())
        outputs = _sanitize_params(problem.outputs, InOrOut.OUT, names, domains)

    normalized_in_data.columns = [i.name for i in inputs]
    normalized_out_data.columns = [o.name for o in outputs]
    normalized_data = pd.concat([normalized_in_data, normalized_out_data], axis=1)
    normalized_data.reset_index(inplace=True)
    objectives = Objectives(
        [
            _sanitize_objective(i, obj, s_output, p_output, problem.data)
            for i, (obj, s_output, p_output) in enumerate(
                zip(problem.objectives, outputs, problem.outputs)
            )
        ]
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
