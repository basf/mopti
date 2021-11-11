from typing import Callable, List

import pandas as pd
from scipy.stats import norm

from opti.parameter import Parameters
from opti.problem import Problem


def _add_noise_to_data(
    df: pd.DataFrame, noisifiers: List[Callable], outputs: Parameters
) -> pd.DataFrame:
    return pd.concat(
        [
            output.round(noisify(df[output.name]))
            for output, noisify in zip(outputs, noisifiers)
        ],
        axis=1,
    )


def noisify_problem(
    problem: Problem,
    noisifiers: List[Callable],
) -> Problem:
    """Creates a new problem that is based on the given one plus noise on the outputs.

    Args:
        problem: given problem where we will add noise
        noisifiers: list of functions that add noise to the outputs

    Returns: new problem with noise on the output
    """

    def noisy_f(X):
        return _add_noise_to_data(problem.f(X), noisifiers, problem.outputs)

    if problem.data is not None:
        data = problem.get_data()
        X = data[problem.inputs.names]
        Yn = _add_noise_to_data(
            data[problem.outputs.names], noisifiers, problem.outputs
        )
        data = pd.concat([X, Yn], axis=1)
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
        name=problem.name,
    )


def noisify_problem_with_gaussian(problem: Problem, mu: float = 0, sigma: float = 0.05):
    """
    Given an instance of a problem, this returns the problem with additive Gaussian noise
    Args:
        problem: problem instance where we add the noise
        mu: mean of the Gaussian noise to be added
        sigma: standard deviation of the Gaussian noise to be added

    Returns: input problem with additive Gaussian noise
    """

    def noisify(y):
        rv = norm(loc=mu, scale=sigma)
        return y + rv.rvs(len(y))

    return noisify_problem(problem, noisifiers=[noisify] * len(problem.outputs))
