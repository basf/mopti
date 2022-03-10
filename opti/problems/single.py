"""Single objective benchmark problems.
"""
import numpy as np
import pandas as pd

from opti.constraint import LinearInequality, NChooseK
from opti.parameter import Categorical, Continuous
from opti.problem import Problem


class Ackley(Problem):
    """Ackley benchmark problem."""

    def __init__(self, n_inputs=2):
        super().__init__(
            name="Ackley problem",
            inputs=[Continuous(f"x{i+1}", [-32.768, +32.768]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        a = 20
        b = 1 / 5
        c = 2 * np.pi
        n = self.n_inputs
        x = self.get_X(X)
        part1 = -a * np.exp(-b * np.sqrt((1 / n) * np.sum(x**2, axis=-1)))
        part2 = -np.exp((1 / n) * np.sum(np.cos(c * x), axis=-1))
        y = part1 + part2 + a + np.exp(1)
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Branin(Problem):
    """The Branin (Branin-Hoo) benchmark problem.

    f(x) = a(x2 - b x1^2 + cx1 - r)^2 + s(1 - t) cos(x1) + s
    a = 1, b = 5.1 / (4 pi^2), c = 5 / pi, r = 6, s = 10 and t = 1 / (8pi)

    It has 3 global optima.
    """

    def __init__(self):
        super().__init__(
            name="Branin function",
            inputs=[Continuous("x1", [-5, 10]), Continuous("x2", [0, 15])],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x1, x2 = self.get_X(X).T
        y = (
            (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
            + 10
        )
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [-np.pi, 12.275, 0.397887],
                [np.pi, 2.275, 0.397887],
                [9.42478, 2.475, 0.397887],
            ],
            columns=self.inputs.names + self.outputs.names,
        )


class Himmelblau(Problem):
    """Himmelblau benchmark problem"""

    def __init__(self):
        super().__init__(
            name="Himmelblau function",
            inputs=[Continuous(f"x{i+1}", [-6, 6]) for i in range(2)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x0, x1 = self.get_X(X).T
        y = (x0**2 + x1 - 11) ** 2 + (x0 + x1**2 - 7) ** 2
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )
        y = np.zeros(4)
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Michalewicz(Problem):
    """Michalewicz benchmark problem.

    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m (m=10 is used here) defines the steepness of they valleys and a larger m leads to a more difficult search.
    """

    def __init__(self, n_inputs: int = 2):
        super().__init__(
            name="Michalewicz function",
            inputs=[Continuous(f"x{i+1}", [0, np.pi]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self.get_X(X)
        m = 10
        i = np.arange(1, self.n_inputs + 1)
        y = -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m), axis=1)
        return pd.DataFrame({"y": y}, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = pd.DataFrame([[2.2, 1.57]], columns=self.inputs.names)
        return pd.concat([x, self.f(x)], axis=1)


class Rosenbrock(Problem):
    """Rosenbrock benchmark problem."""

    def __init__(self, n_inputs=2):
        super().__init__(
            name="Rosenbrock function",
            inputs=[Continuous(f"x{i+1}", [-2.048, 2.048]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self.get_X(X).T
        y = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2, axis=0)
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.ones((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Schwefel(Problem):
    """Schwefel benchmark problem"""

    def __init__(self, n_inputs=2):
        super().__init__(
            name="Schwefel function",
            inputs=[Continuous(f"x{i+1}", [-500, 500]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self.get_X(X)
        y = 418.9829 * self.n_inputs - np.sum(x * np.sin(np.abs(x) ** 0.5), axis=1)
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.full((1, self.n_inputs), 420.9687)
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Sphere(Problem):
    """Sphere benchmark problem."""

    def __init__(self, n_inputs=10):
        super().__init__(
            name="Sphere function",
            inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous("y", [0, 2])],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self.get_X(X)
        y = np.sum((x - 0.5) ** 2, axis=1)
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.full((1, self.n_inputs), 0.5)
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Rastrigin(Problem):
    """Rastrigin benchmark problem."""

    def __init__(self, n_inputs=2):
        super().__init__(
            name="Rastrigin function",
            inputs=[Continuous(f"x{i+1}", [-5, 5]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = self.get_X(X)
        a = 10
        y = a * self.n_inputs + np.sum(x**2 - a * np.cos(2 * np.pi * x), axis=1)
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class ThreeHumpCamel(Problem):
    """Three-hump camel benchmark problem."""

    def __init__(self):
        super().__init__(
            name="Three-hump camel function",
            inputs=[Continuous(f"x{i+1}", [-5, 5]) for i in range(2)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x1, x2 = self.get_X(X).T
        y = 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2
        return pd.DataFrame(y, columns=["y"], index=X.index)

    def get_optima(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.zeros((1, 3)), columns=self.inputs.names + self.outputs.names
        )


class Zakharov(Problem):
    """Zakharov benchmark problem."""

    def __init__(self, n_inputs=2):
        super().__init__(
            name="Zakharov function",
            inputs=[Continuous(f"x{i+1}", [-10, 10]) for i in range(n_inputs)],
            outputs=[Continuous("y")],
        )

    def f(self, X: pd.DataFrame):
        x = self.get_X(X)
        a = 0.5 * np.sum(np.arange(1, self.n_inputs + 1) * x, axis=1)
        y = np.sum(x**2, axis=1) + a**2 + a**4
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Zakharov_NChooseKConstraint(Problem):
    """Zakharov problem with an n-choose-k constraint"""

    def __init__(self, n_inputs=5, n_max_active=3):
        base = Zakharov(n_inputs)
        self.base = base
        super().__init__(
            name="Zakharov with n-choose-k constraint",
            inputs=base.inputs,
            outputs=base.outputs,
            constraints=[NChooseK(names=base.inputs.names, max_active=n_max_active)],
            f=base.f,
        )

    def get_optima(self):
        return self.base.get_optima()


class Zakharov_Constrained(Problem):
    """Zakharov problem with one linear constraint"""

    def __init__(self, n_inputs=5):
        base = Zakharov(n_inputs)
        self.base = base
        super().__init__(
            name="Zakharov with one linear constraint",
            inputs=base.inputs,
            outputs=base.outputs,
            constraints=[LinearInequality(base.inputs.names, lhs=1, rhs=10)],
            f=base.f,
        )

    def get_optima(self):
        return self.base.get_optima()


class Zakharov_Categorical(Problem):
    """Zakharov problem with one categorical input"""

    def __init__(self, n_inputs=3):
        base = Zakharov(n_inputs)
        super().__init__(
            name="Zakharov function with one categorical input",
            inputs=[Continuous(f"x{i}", [-10, 10]) for i in range(n_inputs - 1)]
            + [Categorical("expon_switch", ["one", "two"])],
            outputs=base.outputs,
        )

    def f(self, X: pd.DataFrame):
        x_conti = X[self.inputs.names[:-1]].values  # just the continuous inputs
        a = 0.5 * np.sum(np.arange(1, self.n_inputs) * x_conti, axis=1)
        powers = np.repeat(np.expand_dims([2.0, 2.0, 4.0], 0), repeats=len(X), axis=0)
        modify_powers = X[self.inputs.names[-1]] == "two"
        powers[modify_powers, :] += powers[modify_powers, :]
        res = (
            np.sum(x_conti ** np.expand_dims(powers[:, 0], 1), axis=1)
            + a ** np.expand_dims(powers[:, 1], 0)
            + a ** np.expand_dims(powers[:, 2], 0)
        )
        res_float_array = np.array(res, dtype=np.float64).ravel()
        y = res_float_array
        return pd.DataFrame(y, columns=self.outputs.names, index=X.index)

    def get_optima(self) -> pd.DataFrame:
        x = list(np.zeros(self.n_inputs - 1)) + ["one"]
        y = [0]
        return pd.DataFrame([x + y], columns=self.inputs.names + self.outputs.names)
