from typing import Callable, Optional, Sequence

import numpy as np

from opti import Problem


def create_multi_fidelity(
    problem: Problem,
    fidelities: Sequence[Callable[[np.ndarray], np.ndarray]],
    predicates: Sequence[Callable[[int, np.ndarray], bool]],
    name: Optional[str] = None,
):
    """
    Based on a given problem, this creates a new problem with different fidelities.
    Internally, a counter is increased on every call.
    Args:
        problem: base problem
        fidelities: callables that evaluate the problem with different fidelities
        predicates: callables that decide which fidelity is active at the current call
                    depending on the counter and the inputs
        name: name of the new multi-fidelity instance

    Returns: new problem with different fidelities

    Example:

        # We apply two Gaussians as two fidelities to the zdt1-problem.

        zdt1 = opti.problems.ZDT1(n_inputs=n_inputs)
        first_fidelity = noisify_problem_with_gaussian(zdt1, mu=0, sigma=0.01)
        second_fidelity = noisify_problem_with_gaussian(zdt1, mu=0.1, sigma=0.5)

        def pred_second(counter, X):
            return counter % 3 == 0

        def pred_first(counter, X):
            return not pred_second(counter, _)

        mf_zdt = create_multi_fidelity(
            zdt1, [first_fidelity.f, second_fidelity.f], [pred_first, pred_second]
        )

        [...]

        # when calling subsequently we obtain

        mf_zdt.f(X)  # first fidelity
        mf_zdt.f(X)  # first fidelity
        mf_zdt.f(X)  # second fidelity
        mf_zdt.f(X)  # first fidelity
        mf_zdt.f(X)  # first fidelity
        mf_zdt.f(X)  # second fidelity
        mf_zdt.f(X)  # first fidelity
        ...


    """

    class MultiFidelityF:
        def __init__(self):
            self.counter = 0

        def __call__(self, X):
            self.counter += 1
            for fidelity, pred in zip(fidelities, predicates):
                if pred(self.counter - 1, X):
                    return fidelity(X)
            raise ValueError("no predicate returned true")

    return Problem(
        inputs=problem.inputs,
        outputs=problem.outputs,
        objectives=problem.objectives,
        constraints=problem.constraints,
        output_constraints=problem.output_constraints,
        f=MultiFidelityF(),
        data=None,
        name=problem.name if name is None else name,
    )
