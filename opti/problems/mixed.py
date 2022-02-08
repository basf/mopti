"""
Mixed variables single and multi-objective test problems.
"""

import numpy as np
import pandas as pd

from opti.parameter import Categorical, Continuous, Discrete
from opti.problem import Problem


class DiscreteVLMOP2(Problem):
    """VLMOP2 problem (also known as Fonzeca & Fleming), modified to contain a discrete variable.

    * 2 minimization objectives
    * 1 categorical and n continuous inputs, unconstrained

    See:
        Manson2021, MVMOO: Mixed variable multi-objective optimisation
        https://doi.org/10.1007/s10898-021-01052-9

    """

    def __init__(self, n_inputs: int = 3):
        assert n_inputs >= 2
        super().__init__(
            name="Discrete VLMOP2 test problem",
            inputs=[Categorical("x1", ["a", "b"])]
            + [Continuous(f"x{i+1}", [-2, 2]) for i in range(1, n_inputs)],
            outputs=[Continuous("y1"), Continuous("y2")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X[self.inputs.names[0]].values
        x = X[self.inputs.names[1:]].values
        n = self.n_inputs
        y1 = np.exp(-np.sum((x - n**-0.5) ** 2, axis=1))
        y2 = np.exp(-np.sum((x + n**-0.5) ** 2, axis=1))
        y1 = np.where(d == "a", 1 - y1, 1.25 - y1)
        y2 = np.where(d == "a", 1 - y2, 0.75 - y2)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)


class DiscreteFuelInjector(Problem):
    """Fuel injector test problem, modified to contain an integer variable.

    * 4 objectives,
    * mixed variables, unconstrained

    See:
        Manson2021, MVMOO: Mixed variable multi-objective optimisation
        https://doi.org/10.1007/s10898-021-01052-9
    """

    def __init__(self):
        super().__init__(
            name="Discrete fuel injector test problem",
            inputs=[
                Discrete("x1", [0, 1, 2, 3]),
                Continuous("x2", [-2, 2]),
                Continuous("x3", [-2, 2]),
                Continuous("x4", [-2, 2]),
            ],
            outputs=[Continuous(f"y{i+1}") for i in range(4)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x1 = X["x1"].to_numpy().astype(float)
        x2 = X["x2"].to_numpy().astype(float)
        x3 = X["x3"].to_numpy().astype(float)
        x4 = X["x4"].to_numpy().astype(float)
        x1 *= 0.2
        y1 = (
            0.692
            + 0.4771 * x1
            - 0.687 * x4
            - 0.08 * x3
            - 0.065 * x2
            - 0.167 * x1**2
            - 0.0129 * x1 * x4
            + 0.0796 * x4**2
            - 0.0634 * x1 * x3
            - 0.0257 * x3 * x4
            + 0.0877 * x3**2
            - 0.0521 * x1 * x2
            + 0.00156 * x2 * x4
            + 0.00198 * x2 * x3
            + 0.0184 * x2**2
        )
        y2 = (
            0.37
            - 0.205 * x1
            + 0.0307 * x4
            + 0.108 * x3
            + 1.019 * x2
            - 0.135 * x1**2
            + 0.0141 * x1 * x4
            + 0.0998 * x4**2
            + 0.208 * x1 * x3
            - 0.0301 * x3 * x4
            - 0.226 * x3**2
            + 0.353 * x1 * x2
            - 0.0497 * x2 * x3
            - 0.423 * x2**2
            + 0.202 * x1**2 * x4
            - 0.281 * x1**2 * x3
            - 0.342 * x1 * x4**2
            - 0.245 * x3 * x4**2
            + 0.281 * x3**2 * x4
            - 0.184 * x1 * x2**2
            + 0.281 * x1 * x3 * x4
        )
        y3 = (
            0.153
            - 0.322 * x1
            + 0.396 * x4
            + 0.424 * x3
            + 0.0226 * x2
            + 0.175 * x1**2
            + 0.0185 * x1 * x4
            - 0.0701 * x4**2
            - 0.251 * x1 * x3
            + 0.179 * x3 * x4
            + 0.015 * x3**2
            + 0.0134 * x1 * x2
            + 0.0296 * x2 * x4
            + 0.0752 * x2 * x3
            + 0.0192 * x2**2
        )
        y4 = (
            0.758
            + 0.358 * x1
            - 0.807 * x4
            + 0.0925 * x3
            - 0.0468 * x2
            - 0.172 * x1**2
            + 0.0106 * x1 * x4
            + 0.0697 * x4**2
            - 0.146 * x1 * x3
            - 0.0416 * x3 * x4
            + 0.102 * x3**2
            - 0.0694 * x1 * x2
            - 0.00503 * x2 * x4
            + 0.0151 * x2 * x3
            + 0.0173 * x2**2
        )
        return pd.DataFrame({"y1": y1, "y2": y2, "y3": y3, "y4": y4})
