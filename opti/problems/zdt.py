"""
ZDT benchmark problem suite.
All problems are bi-objective, have D continuous inputs and are unconstrained.

Zitzler, Deb, Thiele 2000 - Comparison of Multiobjective Evolutionary Algorithms: Empirical Results
http://dx.doi.org/10.1162/106365600568202
"""
import numpy as np
import pandas as pd

from opti.parameter import Continuous
from opti.problem import Problem


class ZDT1(Problem):
    """ZDT-1 benchmark problem."""

    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT-1 problem",
            inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i+1}", [0, np.inf]) for i in range(2)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names[1:]].to_numpy()
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x, axis=1)
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - (y1 / g) ** 0.5)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT2(Problem):
    """ZDT-2 benchmark problem."""

    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT-2 problem",
            inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i+1}", [0, np.inf]) for i in range(2)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names[1:]].to_numpy()
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x, axis=1)
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - (y1 / g) ** 2)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.power(x, 2)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT3(Problem):
    """ZDT-3 benchmark problem."""

    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT-3 problem",
            inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i+1}", [-np.inf, np.inf]) for i in range(2)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names[1:]].to_numpy()
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x, axis=1)
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - (y1 / g) ** 0.5 - (y1 / g) * np.sin(10 * np.pi * y1))
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        regions = [
            [0, 0.0830015349],
            [0.182228780, 0.2577623634],
            [0.4093136748, 0.4538821041],
            [0.6183967944, 0.6525117038],
            [0.8233317983, 0.8518328654],
        ]

        pf = []
        for r in regions:
            x1 = np.linspace(r[0], r[1], int(points / len(regions)))
            x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
            pf.append(np.stack([x1, x2], axis=1))

        y = np.concatenate(pf, axis=0)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT4(Problem):
    """ZDT-4 benchmark problem."""

    def __init__(self, n_inputs=10):
        super().__init__(
            name="ZDT-4 problem",
            inputs=[Continuous("x1", [0, 1])]
            + [Continuous(f"x{i+1}", [-5, 5]) for i in range(1, n_inputs)],
            outputs=[Continuous(f"y{i+1}", [0, np.inf]) for i in range(2)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names].to_numpy()
        g = 1 + 10 * (self.n_inputs - 1)
        for i in range(1, self.n_inputs):
            g += x[:, i] ** 2 - 10 * np.cos(4.0 * np.pi * x[:, i])
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - np.sqrt(y1 / g))
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT6(Problem):
    """ZDT-6 benchmark problem."""

    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT-6 problem",
            inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i+1}", [-np.inf, np.inf]) for i in range(2)],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.inputs.names].to_numpy()
        n = self.n_inputs
        g = 1 + 9 * (np.sum(x[:, 1:], axis=1) / (n - 1)) ** 0.25
        y1 = 1 - np.exp(-4 * x[:, 0]) * (np.sin(6 * np.pi * x[:, 0])) ** 6
        y2 = g * (1 - (y1 / g) ** 2)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0.2807753191, 1, points)
        y = np.stack([x, 1 - x**2], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)
