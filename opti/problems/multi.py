from itertools import product
from typing import Optional, Union

import numpy as np
import pandas as pd

import opti
from opti.constraint import LinearInequality, NonlinearInequality
from opti.parameter import Continuous
from opti.problem import Problem


class Qapi1(Problem):
    """Constrained problem from the Qriteria API tests.
    Note that while the function is convex, the constraints are not.

    minimize
        f1(x) = (x1 - 2)^2 + (x2 - 1)^2
        f2(x) = x1^2 + (x2 - 3)^2
    for
        x1 in [0, inf)
        x2 in (-inf, inf)
    s.t.
        0 <= x1
        c1(x) = - x1^2 + x2 <= 0
        c2(x) = - x1 - x2 + 2 <= 0

    The ideal point is [0, 0] and the nadir is [8, 8].
    """

    def __init__(self):
        super().__init__(
            name="Constrained bi-objective problem",
            inputs=[Continuous("x1", [0, 10]), Continuous("x2", [-10, 10])],
            outputs=[Continuous("y1"), Continuous("y2")],
            constraints=[
                NonlinearInequality("x2 - x1**2"),
                NonlinearInequality("2 - x1 - x2"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y1": X.eval("(x1 - 2)**2 + (x2 - 1)**2"),
                "y2": X.eval("x1**2 + (x2 - 3)**2"),
            }
        )


class Hyperellipsoid(Problem):
    """Hyperellipsoid in n dimensions

    minimize
        f_m(x) = x_m    m = 1, ... n
    for
        x in R^n
    s.t.
        sum((x / a)^2) - 1 <= 0

    The ideal point is -a and the is nadir 0^n.

    Args:
        n (int, optional): Dimension of the hyperellipsoid. Defaults to 5.
        a (list-like, optional): Half length of principal axes. a = None or a = [1, ...] results in a hypersphere.
    """

    def __init__(self, n: int = 5, a: Optional[Union[list, np.ndarray]] = None):
        if a is None:
            a = np.ones(n)
            constr = " + ".join([f"x{i+1}**2" for i in range(n)]) + " - 1"
        else:
            a = np.array(a).squeeze()
            if len(a) != n:
                raise ValueError("Dimension of half axes doesn't match input dimension")
            constr = " + ".join([f"(x{i+1}/{a[i]})**2" for i in range(n)]) + " - 1"
        self.a = a

        super().__init__(
            name="Hyperellipsoid",
            inputs=[Continuous(f"x{i+1}", [-a[i], a[i]]) for i in range(n)],
            outputs=[Continuous(f"y{i+1}", [-a[i], a[i]]) for i in range(n)],
            constraints=[NonlinearInequality(constr)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        y = X[self.inputs.names]
        y.columns = self.outputs.names
        return y

    def get_optima(self, n=10) -> pd.DataFrame:
        X = opti.sampling.sphere.sample(self.n_inputs, n, positive=True)
        X = np.concatenate([-np.eye(self.n_inputs), -X], axis=0)[:n]
        Y = self.a * X
        return pd.DataFrame(
            data=np.column_stack([X, Y]),
            columns=self.inputs.names + self.outputs.names,
        )


class Daechert1(Problem):
    """Problem with a non-convex Pareto front.

    From Dächert & Teichert 2020, An improved hyperboxing algorithm for calculating a Pareto front representation, https://arxiv.org/abs/2003.14249

    The ideal point is [-1.37, -1.61, -4] and the nadir is [0, 0, -1.44].
    """

    def __init__(self):
        super().__init__(
            name="Daechert-1",
            inputs=[
                Continuous("x1", domain=[0, np.pi]),
                Continuous("x2", domain=[0, 10]),
                Continuous("x3", domain=[1.2, 10]),
            ],
            outputs=[Continuous(f"y{i+1}") for i in range(3)],
            constraints=[NonlinearInequality("- cos(x1) - exp(-x2) + x3")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"y1": -X["x1"], "y2": -X["x2"], "y3": -X["x3"] ** 2})


class Daechert2(Problem):
    """Unconstrained problem with a Pareto front resembling a comet.

    From Dächert & Teichert 2020, An improved hyperboxing algorithm for calculating a Pareto front representation, https://arxiv.org/abs/2003.14249

    minimize
        f1(x) = (1 + x3) (x1^3 x2^2 - 10 x1 - 4 x2)
        f2(x) = (1 + x3) (x1^3 x2^2 - 10 x1 + 4 x2)
        f3(x) = 3 (1 + x3) x1^2
    s.t.
         1 <= x1 <= 3.5
        -2 <= x2 <= 2
         0 <= x3 <= 1

    The ideal point is [-70.19, -70.19, 3] and the nadir is [4, 4, 73.5].
    """

    def __init__(self):
        super().__init__(
            name="Daechert-2",
            inputs=[
                Continuous("x1", domain=[1, 3.5]),
                Continuous("x2", domain=[-2, 2]),
                Continuous("x3", domain=[0, 1]),
            ],
            outputs=[Continuous(f"y{i + 1}") for i in range(3)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y1": X.eval("(1 + x3) * (x1**3 * x2**2 - 10 * x1 - 4 * x2)"),
                "y2": X.eval("(1 + x3) * (x1**3 * x2**2 - 10 * x1 + 4 * x2)"),
                "y3": X.eval("3 * (1 + x3) * x1**2"),
            }
        )


class Daechert3(Problem):
    """Modification of DTLZ7, with a Pareto consisting of 4 disconnected parts.

    From Dächert & Teichert 2020, An improved hyperboxing algorithm for calculating a Pareto front representation, https://arxiv.org/abs/2003.14249

    The ideal point is [0, 0, 2.61] and the nadir is [0.86, 0.86, 6].
    """

    def __init__(self):
        super().__init__(
            name="Daechert-3",
            inputs=[Continuous(f"x{i+1}", domain=[0, 1]) for i in range(2)],
            outputs=[Continuous(f"y{i+1}") for i in range(3)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names]
        return pd.DataFrame(
            {
                "y1": X["x1"],
                "y2": X["x2"],
                "y3": 6 - np.sum(x * (1 + np.sin(3 * np.pi * x)), axis=1),
            }
        )


class OmniTest(Problem):
    """Bi-objective benchmark problem with D inputs and a multi-modal Pareto set.

    It has 3^D Pareto subsets in the decision space corresponding to the same Pareto front.

    Reference:
        Deb & Tiwari "Omni-optimizer: A generic evolutionary algorithm for single and multi-objective optimization"
    """

    def __init__(self, n_inputs: int = 2):
        super().__init__(
            name="Omni",
            inputs=[Continuous(f"x{i+1}", domain=[0, 6]) for i in range(2)],
            outputs=[Continuous("y1"), Continuous("y2")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.inputs.names]
        return pd.DataFrame(
            {
                "y1": np.sum(np.sin(np.pi * X), axis=1),
                "y2": np.sum(np.cos(np.pi * X), axis=1),
            }
        )

    def get_optima(self) -> pd.DataFrame:
        n = 11  # points per set (3^D sets)
        s = [np.linspace(1, 1.5, n) + 2 * i for i in range(3)]
        C = list(
            product(
                *[
                    s,
                ]
                * self.n_inputs
            )
        )
        C = np.moveaxis(C, 1, 2).reshape(-1, 2)
        X = pd.DataFrame(C, columns=self.inputs.names)
        XY = pd.concat([X, self.f(X)], axis=1)
        XY["_patch"] = np.repeat(np.arange(3**self.n_inputs), n)
        return XY


class Poloni(Problem):
    """Poloni benchmark problem."""

    def __init__(self):
        super().__init__(
            name="Poloni function",
            inputs=[Continuous(f"x{i+1}", [-np.pi, np.pi]) for i in range(2)],
            outputs=[Continuous("y1"), Continuous("y2")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x1, x2 = self.get_X(X).T
        A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        B1 = 0.5 * np.sin(x1) - 2 * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
        B2 = 1.5 * np.sin(x1) - np.cos(x1) + 2 * np.sin(x2) - 0.5 * np.cos(x2)
        return pd.DataFrame(
            {
                "y1": 1 + (A1 - B1) ** 2 + (A2 - B2) ** 2,
                "y2": (x1 + 3) ** 2 + (x2 + 1) ** 2,
            },
            index=X.index,
        )


class WeldedBeam(Problem):
    """Design optimization of a welded beam.

    This is a bi-objective problem with 4 inputs and 3 (non-)linear inequality constraints.
    The two objectives are the fabrication cost of the beam and the deflection of the end of the beam under the applied load P.
    The load P is fixed at 6000 lbs, and the distance L is fixed at 14 inch.

    Note that for simplicity the constraint shear stress < 13600 psi is not included.

    See https://www.mathworks.com/help/gads/multiobjective-optimization-welded-beam.html
    """

    def __init__(self):
        super().__init__(
            name="Welded beam problem",
            inputs=[
                Continuous("h", [0.125, 5]),  # thickness of welds
                Continuous("l", [0.1, 10]),  # length of welds
                Continuous("t", [0.1, 10]),  # height of beam
                Continuous("b", [0.125, 5]),  # width of beam
            ],
            outputs=[Continuous("cost"), Continuous("deflection")],
            constraints=[
                # h <= b, weld thickness cannot exceed beam width
                LinearInequality(["h", "b"], lhs=[1, -1], rhs=0),
                # normal stress on the welds cannot exceed 30000 psi
                NonlinearInequality("6000 * 6 * 14 / b / t**3 - 30000"),
                # buckling load capacity must exceed 6000 lbs
                NonlinearInequality(
                    "6000 - 60746.022 * (1 - 0.0282346 * t) * t * b**4"
                ),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x1, x2, x3, x4 = self.get_X(X).T
        return pd.DataFrame(
            {
                "cost": 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2),
                "deflection": 2.1952 / (x4 * x3**3),
            },
            index=X.index,
        )
