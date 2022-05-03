import numpy as np
import pandas as pd

import opti
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
    constr2(X).values

    # objectives
    objectives = Objectives(
        [Minimize("y1"), Maximize("y2"), CloseToTarget("y3", target=7)]
    )
    Y = pd.DataFrame({"y1": [1, 2, 3], "y2": [7, 4, 5], "y3": [5, 6.9, 12]})
    objectives(Y)

    # problem
    Problem(
        inputs=inputs, outputs=outputs, constraints=constraints, objectives=objectives
    )


def test_problem_reduction():
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0.1, 0.7]),
            opti.Continuous("x2", [0, 0.8]),
            opti.Continuous("x3", [0.3, 0.9]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(["x1", "x2", "x3"], rhs=1)],
    )

    reduced_problem, transform = opti.tools.reduce_problem(problem)
    print(reduced_problem)

    X1 = problem.sample_inputs(10)
    Xr = transform.drop_data(X1)
    X2 = transform.augment_data(Xr)
    assert np.allclose(X1, X2[X1.columns])

    def f(X):
        y = X[["A1", "A2", "A3", "A4"]] @ [1, -2, 3, 2]
        y += X[["B1", "B2", "B3"]] @ [0.1, 0.4, 0.3]
        y += X["Temperature"] / 30
        y += X["Process"] == "process 2"
        return pd.DataFrame({"y": y})

    problem = opti.Problem(
        inputs=[
            opti.Continuous("A1", [0, 0.9]),
            opti.Continuous("A2", [0, 0.8]),
            opti.Continuous("A3", [0, 0.9]),
            opti.Continuous("A4", [0, 0.9]),
            opti.Continuous("B1", [0.3, 0.9]),
            opti.Continuous("B2", [0, 0.8]),
            opti.Continuous("B3", [0.1, 1]),
            opti.Discrete("Temperature", [20, 25, 30]),
            opti.Categorical("Process", ["process 1", "process 2", "process 3"]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(["A1", "A2", "A3", "A4"], rhs=1),
            opti.LinearEquality(["B1", "B2", "B3"], rhs=1),
            opti.LinearInequality(["A1", "A2"], lhs=[1, 2], rhs=0.8),
        ],
        f=f,
    )

    reduced_problem, transform = opti.tools.reduce_problem(problem)
    print(reduced_problem)

    Xr = reduced_problem.sample_inputs(10)
    X = transform.augment_data(Xr)
    y1 = problem.f(X)
    y2 = reduced_problem.f(Xr)
    assert np.allclose(y1, y2)
