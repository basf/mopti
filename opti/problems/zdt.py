"""
ZDT problem test suite.
Reference: Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. “Comparison of multiobjective evolutionary algorithms: Empirical results.” Evolutionary computation 8.2 (2000): 173-195. doi: 10.1.1.30.5848
Adapted from https://github.com/msu-coinlab/pymoo/blob/master/pymoo/problems/multi/zdt.py
"""
import numpy as np
import pandas as pd

from opti.parameter import Continuous
from opti.problem import Problem


class ZDT1(Problem):
    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT1 function",
            inputs=[Continuous(f"x{i}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i}", [0, np.inf]) for i in range(2)],
        )

    def f(self, x):
        f1 = x[:, 0]
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - (f1 / g) ** 0.5)
        return np.stack([f1, f2], axis=1)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT2(Problem):
    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT2 function",
            inputs=[Continuous(f"x{i}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i}", [0, np.inf]) for i in range(2)],
        )

    def f(self, x):
        f1 = x[:, 0]
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.stack([f1, f2], axis=1)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.power(x, 2)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT3(Problem):
    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT3 function",
            inputs=[Continuous(f"x{i}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i}", [-np.inf, np.inf]) for i in range(2)],
        )

    def f(self, x):
        f1 = x[:, 0]
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - (f1 / g) ** 0.5 - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.stack([f1, f2], axis=1)

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
    def __init__(self, n_inputs=10):
        super().__init__(
            name="ZDT4 function",
            inputs=[Continuous(f"x{0}", [0, 1])]
            + [Continuous(f"x{i}", [-5, 5]) for i in range(1, n_inputs)],
            outputs=[Continuous(f"y{i}", [0, np.inf]) for i in range(2)],
        )

    def f(self, x):
        f1 = x[:, 0]
        g = 1 + 10 * (self.n_inputs - 1)
        for i in range(1, self.n_inputs):
            g += x[:, i] ** 2 - 10 * np.cos(4.0 * np.pi * x[:, i])
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.stack([f1, f2], axis=1)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)


class ZDT6(Problem):
    def __init__(self, n_inputs=30):
        super().__init__(
            name="ZDT6 function",
            inputs=[Continuous(f"x{i}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous(f"y{i}", [-np.inf, np.inf]) for i in range(2)],
        )

    def f(self, x):
        n = self.n_inputs
        f1 = 1 - np.exp(-4 * x[:, 0]) * (np.sin(6 * np.pi * x[:, 0])) ** 6
        g = 1 + 9 * (np.sum(x[:, 1:], axis=1) / (n - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return np.stack([f1, f2], axis=1)

    def get_optima(self, points=100):
        x = np.linspace(0.2807753191, 1, points)
        y = np.stack([x, 1 - x ** 2], axis=1)
        return pd.DataFrame(y, columns=self.outputs.names)
