"""
Simple 1D problems for visually assessing probabilistic surrogate models.
Note: these problems should be output-noisified, e.g.
```
import opti

problem = opti.problems.noisify_problem_with_gaussian(
    opti.problems.Line1D(),
    sigma=0.1
)
```
"""
import numpy as np
import pandas as pd

from opti.parameter import Continuous, Discrete
from opti.problem import Problem


class Line1D(Problem):
    def __init__(self):
        def f(x):
            return 0.1 * x + 1

        x = np.array([0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5])
        data = pd.DataFrame({"x": x, "y": f(x)})

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=data,
        )


class Parabola1D(Problem):
    def __init__(self):
        def f(x):
            return 0.025 * (x - 5) ** 2 + 1

        x = np.array([0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5])
        data = pd.DataFrame({"x": x, "y": f(x)})

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=data,
        )


class Sinus1D(Problem):
    def __init__(self):
        def f(x):
            return np.sin(x * 2 * np.pi / 10) / 2 + 2

        x = np.array([0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5])
        data = pd.DataFrame({"x": x, "y": f(x)})

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=data,
        )


class Sigmoid1D(Problem):
    def __init__(self):
        def f(x):
            return 1 / (1 + np.exp(-(x - 5) * 2)) + 1

        x = np.array([0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5])
        data = pd.DataFrame({"x": x, "y": f(x)})

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Continuous("y", [0, 3])],
            f=f,
            data=data,
        )


class Step1D(Problem):
    def __init__(self):
        def f(x):
            return (x > 1.1).astype(float)

        x = np.array([0.2, 1, 1.5, 2, 3, 3.1, 6.5, 7, 7.2, 7.5])
        data = pd.DataFrame({"x": x, "y": f(x)})

        super().__init__(
            inputs=[Continuous("x", [0, 10])],
            outputs=[Discrete("y", [0, 1])],
            f=f,
            data=data,
        )
