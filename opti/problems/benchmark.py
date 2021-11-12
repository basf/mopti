import numpy as np
import pandas as pd

import opti
from opti.constraint import NonlinearInequality
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

    def __init__(self, n=5, a=None):
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
        x = X[self.inputs.names].values
        return pd.DataFrame(
            {
                "y1": X["x1"],
                "y2": X["x2"],
                "y3": 6 - np.sum(x * (1 + np.sin(3 * np.pi * x)), axis=1),
            }
        )
