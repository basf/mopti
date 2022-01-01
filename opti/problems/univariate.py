"""
Simple 1D problems for assessing probabilistic surrogate models.
Note: these problems should be output-noisified, e.g.
```
import opti

problem = opti.problems.noisify_problem_with_gaussian(
    opti.problems.Line1D(),
    sigma=0.1
)
```
"""
import pandas as pd

from opti.parameter import Continuous, Discrete
from opti.problem import Problem

_X = pd.DataFrame({"x": [0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5]})


class Line1D(Problem):
    """A line."""

    def __init__(self):
        def f(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"y": X.eval("0.1 * x + 1")}, index=X.index)

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=pd.concat([_X, f(_X)], axis=1),
        )


class Parabola1D(Problem):
    """A parabola."""

    def __init__(self):
        def f(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"y": X.eval("0.025 * (x - 5) ** 2 + 1")}, index=X.index
            )

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=pd.concat([_X, f(_X)], axis=1),
        )


class Sinus1D(Problem):
    """A sinus-function with one full period over the domain."""

    def __init__(self):
        def f(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"y": X.eval("sin(x * 2 * 3.14159 / 10) / 2 + 2")}, index=X.index
            )

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=pd.concat([_X, f(_X)], axis=1),
        )


class Sigmoid1D(Problem):
    """A smooth step at x=5."""

    def __init__(self):
        def f(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"y": X.eval("1 / (1 + exp(-2 * (x - 5))) + 1")}, index=X.index
            )

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=pd.concat([_X, f(_X)], axis=1),
        )


class Step1D(Problem):
    """A discrete step at x=1.1."""

    def __init__(self):
        def f(X: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"y": X.eval("x > 1.1").astype(float)}, index=X.index)

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Discrete("y", [0, 1])],
            f=f,
            data=pd.concat([_X, f(_X)], axis=1),
        )
