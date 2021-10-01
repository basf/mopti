from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from opti.parameter import Parameters
from opti.problem import Problem


def _add_noise_to_data(Y, noisifiers, outputs: Parameters):
    Y = np.atleast_2d(Y)
    noisy_Y = np.stack(
        [noisify(Y[:, col]) for noisify, col in zip(noisifiers, range(Y.shape[1]))],
        axis=1,
    )
    for col, output in enumerate(outputs):
        noisy_Y[:, col] = output.round(noisy_Y[:, col])
    return noisy_Y


def noisify_problem(
    problem: Problem,
    noisifiers: Union[Callable, List[Callable]],
    name: Optional[str] = None,
) -> Problem:
    """Creates a new problem that is based on the given one plus noise on the outputs.

    Args:
        problem: given problem where we will add noise
        noisifiers: function or list of functions that add noise to the outputs
        name: name of the new problem

    Returns: new problem with noise on the output
    """

    def noisy_f(X):
        nonlocal noisifiers
        if isinstance(noisifiers, Callable):
            noisifiers = [noisifiers] * problem.n_outputs
        Y = problem.f(X)

        return _add_noise_to_data(Y, noisifiers, problem.outputs)

    if problem.data is not None:
        noisy_Y = _add_noise_to_data(
            problem.data[problem.outputs.names].values, noisifiers, problem.outputs
        )
        data = pd.concat(
            [
                problem.data[problem.inputs.names],
                pd.DataFrame(
                    columns=problem.outputs.names,
                    index=problem.data.index,
                    data=noisy_Y,
                ),
            ],
            axis=1,
        )
    else:
        data = None

    return Problem(
        inputs=problem.inputs,
        outputs=problem.outputs,
        objectives=problem.objectives,
        constraints=problem.constraints,
        output_constraints=problem.output_constraints,
        f=noisy_f,
        data=data,
        name=problem.name if name is None else name,
    )


class NoiseType(Enum):
    ADDITIVE = 1
    MULTIPLICATIVE = 2


def noisify_problem_with_scipy_stats(
    problem: Problem,
    random_variables: Union[list, Any],
    rv_kwargs: Union[None, dict, List[dict]] = None,
    noise_types: Union[List[NoiseType], NoiseType] = NoiseType.ADDITIVE,
    name: Optional[str] = None,
) -> Problem:
    """
    Add additive or multiplicative noise drawn from a distribution in scipy.stats
    https://docs.scipy.org/doc/scipy/reference/stats.html
    Args:
        problem: problem instance where we want to add noise to
        random_variables: list of instances or instance of random variables, e.g.,
                          [scipy.stats.beta, scipy.stats.cauchy], each element of the
                          list is applied to the corresponding data column
        rv_kwargs: keyword arguments for the random variables, type must correspond to
                   the on of random_variables if given.
        noise_types: list of the same length as random_variables or single value,
                     decides whether the noise is added or multiplied
        name: name of the noisy problem

    Returns: noisy problem instance

    Example:
        To see how to obtain noisy flow-reactor-problems check the following examples.

        1.) Additive Gaussian noise for all output columns

        from scipy import stats
        from opti.problems.flow_reactor_unconstrained import get_problem_pareto
        flow = get_problem_pareto()
        rv = stats.norm
        noisy_flow = noisify_problem_with_scipy_stats(flow, rv)

        2.) Additive Gaussian noise for all output columns with non-standard parameters

        from scipy import stats
        from opti.problems.flow_reactor_unconstrained import get_problem_pareto
        flow = get_problem_pareto()
        rv = stats.norm
        rv_kwargs = {"loc": 0.5, "scale": 0.3}
        noisy_flow = noisify_problem_with_scipy_stats(flow, rv, rv_kwargs=rv_kwargs)

        3.) Different noise distributions and types for the different output columns

        from scipy import stats
        from opti.problems.flow_reactor_unconstrained import get_problem_pareto
        flow = get_problem_pareto()
        rvs = [stats.truncnorm, stats.norm, stats.gamma]
        rv_kwargs = [{"a": 0.1, "b": 2}, {"loc": 0.5, "scale": 0.3}, {"a": 1.99}]
        noise_types = [NoiseType.ADDITIVE, NoiseType.ADDITIVE, NoiseType.MULTIPLICATIVE]
        noisy_flow = noisify_problem_with_scipy_stats(
            flow, rvs, rv_kwargs=rv_kwargs, noise_types=noise_types
        )
    """
    # listify single element inputs
    if not isinstance(random_variables, list):
        random_variables = [random_variables] * problem.n_outputs
        rv_kwargs = [rv_kwargs] if rv_kwargs is not None else [{}]
        rv_kwargs = rv_kwargs * problem.n_outputs
    if not isinstance(noise_types, list):
        noise_types = [noise_types] * problem.n_outputs

    ops = {
        NoiseType.ADDITIVE: lambda x, y: x + y,
        NoiseType.MULTIPLICATIVE: lambda x, y: x * y,
    }
    op_names = {
        NoiseType.ADDITIVE: "add",
        NoiseType.MULTIPLICATIVE: "mul",
    }

    def apply_noise(data_column, noise_type, random_variable, rv_kwargs):
        sampled_noise = random_variable.rvs(size=data_column.size, **rv_kwargs)
        return ops[noise_type](data_column, sampled_noise)

    noisifiers = [
        partial(apply_noise, noise_type=nt, random_variable=rv, rv_kwargs=rvk)
        for nt, rv, rvk in zip(noise_types, random_variables, rv_kwargs)
    ]
    if name is None:
        rv_type_names = [
            f"{rv.name}_{op_names[nt]}" for rv, nt in zip(random_variables, noise_types)
        ]
        name = f"{problem.name} with rvs { ','.join(rv_type_names)}"
    return noisify_problem(problem, noisifiers, name=name)


def noisify_problem_with_gaussian(problem: Problem, mu: float = 0, sigma: float = 0.05):
    """
    Given an instance of a problem, this returns the problem with additive Gaussian noise
    Args:
        problem: problem instance where we add the noise
        mu: mean of the Gaussian noise to be added
        sigma: standard deviation of the Gaussian noise to be added

    Returns: input problem with additive Gaussian noise
    """
    rv_kwargs = {"loc": mu, "scale": sigma}
    noise_type = NoiseType.ADDITIVE
    return noisify_problem_with_scipy_stats(problem, stats.norm, rv_kwargs, noise_type)
