import pandas as pd

from opti import Problem
from opti.constraint import (
    Constraints,
    LinearEquality,
    LinearInequality,
    NChooseK,
    NonlinearEquality,
    NonlinearInequality,
)
from opti.objective import CloseToTarget, Maximize, Minimize, Objectives
from opti.parameter import Categorical, Continuous, Discrete, Parameters


def test_overview():
    # inputs, outputs
    inputs = Parameters(
        [
            Continuous("x1", domain=[0, 1]),
            Continuous("x2", domain=[0, 1]),
            Continuous("x3", domain=[0, 1]),
            Discrete("x4", domain=[1, 2, 5, 7.5]),
            Categorical("x5", domain=["A", "B", "C"]),
        ]
    )

    outputs = Parameters(
        [
            Continuous("y1", domain=[0, None]),
            Continuous("y2", domain=[None, None]),
            Continuous("y3", domain=[0, 100]),
        ]
    )

    X = inputs.sample(5)
    inputs.contains(X)
    inputs.names

    # constraints
    constr1 = LinearEquality(["x1", "x2", "x3"], lhs=[1, 1, 1], rhs=1)
    constr2 = LinearInequality(["x1", "x3"], lhs=[1, 1], rhs=0.8)
    constr3 = NonlinearEquality("x1**2 + x2**2 - 1")
    constr4 = NonlinearInequality("1/x1 + 1/x2 - 2")
    constr5 = NChooseK(["x1", "x2", "x3"], max_active=2)
    constraints = Constraints([constr1, constr2, constr3, constr4, constr5])
    constr2.satisfied(X).values
    constr2.eval(X).values

    # objectives
    objectives = Objectives(
        [Minimize("y1"), Maximize("y2"), CloseToTarget("y3", target=7)]
    )
    Y = pd.DataFrame({"y1": [1, 2, 3], "y2": [7, 4, 5], "y3": [5, 6.9, 12]})
    objectives.eval(Y)

    # problem
    Problem(
        inputs=inputs, outputs=outputs, constraints=constraints, objectives=objectives
    )
