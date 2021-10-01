import numpy as np
import pandas as pd

from opti.constraint import LinearInequality, NChooseK
from opti.parameter import Categorical, Continuous
from opti.problem import Problem


class Ackley(Problem):
    def __init__(self, n_inputs=2):
        super().__init__(
            name="Ackley function",
            inputs=[Continuous(f"x{i}", [-32.768, +32.768]) for i in range(n_inputs)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x):
        a = 20
        b = 1 / 5
        c = 2 * np.pi
        n = self.n_inputs
        x = np.atleast_2d(x)
        part1 = -a * np.exp(-b * np.sqrt((1 / n) * np.sum(x ** 2, axis=-1)))
        part2 = -np.exp((1 / n) * np.sum(np.cos(c * x), axis=-1))
        return part1 + part2 + a + np.exp(1)

    def get_optima(self):
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Himmelblau(Problem):
    def __init__(self):
        super().__init__(
            name="Himmelblau function",
            inputs=[Continuous(f"x{i}", [-6, 6]) for i in range(2)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x):
        x0, x1 = np.atleast_2d(x).T
        return (x0 ** 2 + x1 - 11) ** 2 + (x0 + x1 ** 2 - 7) ** 2

    def get_optima(self):
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


class Rosenbrock(Problem):
    def __init__(self, n_inputs=2):
        super().__init__(
            name="Rosenbrock function",
            inputs=[Continuous(f"x{i}", [-2.048, 2.048]) for i in range(n_inputs)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x):
        x = np.atleast_2d(x).T
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2, axis=0)

    def get_optima(self):
        x = np.ones((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Schwefel(Problem):
    def __init__(self, n_inputs=2):
        super().__init__(
            name="Schwefel function",
            inputs=[Continuous(f"x{i}", [-500, 500]) for i in range(n_inputs)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x):
        x = np.atleast_2d(x)
        return 418.9829 * self.n_inputs - np.sum(x * np.sin(np.abs(x) ** 0.5), axis=1)

    def get_optima(self):
        x = np.full((1, self.n_inputs), 420.9687)
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Sphere(Problem):
    def __init__(self, n_inputs=10):
        super().__init__(
            name="Sphere function",
            inputs=[Continuous(f"x{i}", [0, 1]) for i in range(n_inputs)],
            outputs=[Continuous("y", [0, 2])],
        )

    def f(self, x):
        x = np.atleast_2d(x)
        return np.sum((x - 0.5) ** 2, axis=1)

    def get_optima(self):
        x = np.full((1, self.n_inputs), 0.5)
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Rastrigin(Problem):
    def __init__(self, n_inputs=2):
        super().__init__(
            name="Rastrigin function",
            inputs=[Continuous(f"x{i}", [-5, 5]) for i in range(n_inputs)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x):
        x = np.atleast_2d(x)
        a = 10
        return a * self.n_inputs + np.sum(x ** 2 - a * np.cos(2 * np.pi * x), axis=1)

    def get_optima(self):
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Zakharov(Problem):
    def __init__(self, n_inputs=2):
        super().__init__(
            name="Zakharov function",
            inputs=[Continuous(f"x{i}", [-10, 10]) for i in range(n_inputs)],
            outputs=[Continuous("y", [-np.inf, np.inf])],
        )

    def f(self, x: np.ndarray):
        x = np.atleast_2d(x)
        a = 0.5 * np.sum(np.arange(1, self.n_inputs + 1) * x, axis=1)
        return np.sum(x ** 2, axis=1) + a ** 2 + a ** 4

    def get_optima(self):
        x = np.zeros((1, self.n_inputs))
        y = 0
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)


class Zakharov_NChooseKConstraint(Problem):
    """Variant of the Zakharov problem with an n-choose-k constraint"""

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
    """Variant of the Zakharov problem with one linear constraint"""

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
    """Variant of the Zakharov problem with one categorical input"""

    def __init__(self, n_inputs=3):
        base = Zakharov(n_inputs)
        super().__init__(
            name="Zakharov function with one categorical input",
            inputs=[Continuous(f"x{i}", [-10, 10]) for i in range(n_inputs - 1)]
            + [Categorical("expon_switch", ["one", "two"])],
            outputs=base.outputs,
        )

    def f(self, x: np.ndarray):
        x_conti = np.atleast_2d(x[:, :-1])  # Just the continuous inputs
        a = 0.5 * np.sum(np.arange(1, self.n_inputs) * x_conti, axis=1)
        powers = np.repeat(np.expand_dims([2.0, 2.0, 4.0], 0), repeats=len(x), axis=0)
        modify_powers = x[:, -1] == "two"
        powers[modify_powers, :] += powers[modify_powers, :]
        res = (
            np.sum(x_conti ** np.expand_dims(powers[:, 0], 1), axis=1)
            + a ** np.expand_dims(powers[:, 1], 0)
            + a ** np.expand_dims(powers[:, 2], 0)
        )
        res_float_array = np.array(res, dtype=np.float64).ravel()
        return res_float_array

    def get_optima(self):
        x = list(np.zeros(self.n_inputs - 1)) + ["one"]
        y = [0]
        return pd.DataFrame([x + y], columns=self.inputs.names + self.outputs.names)
