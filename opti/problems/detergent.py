import numpy as np
import pandas as pd

from opti.constraint import LinearInequality, NChooseK
from opti.objective import Maximize
from opti.parameter import Continuous, Discrete
from opti.problem import Problem


def _poly2(x: np.ndarray) -> np.ndarray:
    """Quadratic feature expansion including bias term."""
    return np.concatenate([[1], x, np.outer(x, x)[np.triu_indices(5)]])


class Detergent(Problem):
    """Detergent formulation problem.

    There are 5 outputs representing the washing performance on different stain types.
    Each output is modeled as a second degree polynomial.
    The formulation consists of 5 components.
    The sixth input is a filler (water) and is factored out and it's parameter bounds
    0.6 < water < 0.8 result in 2 linear inequality constraints for the other parameters.
    """

    def __init__(self):
        # coefficients for the 2-order polynomial; generated with
        # base = 3 * np.ones((1, 5))
        # scale = PolynomialFeatures(degree=2).fit_transform(base).T
        # coef = np.random.RandomState(42).normal(scale=scale, size=(len(scale), 5))
        # coef = np.clip(coef, 0, None)
        self.coef = np.array(
            [
                [0.4967, 0.0, 0.6477, 1.523, 0.0],
                [0.0, 4.7376, 2.3023, 0.0, 1.6277],
                [0.0, 0.0, 0.7259, 0.0, 0.0],
                [0.0, 0.0, 0.9427, 0.0, 0.0],
                [4.3969, 0.0, 0.2026, 0.0, 0.0],
                [0.3328, 0.0, 1.1271, 0.0, 0.0],
                [0.0, 16.6705, 0.0, 0.0, 7.4029],
                [0.0, 1.8798, 0.0, 0.0, 1.7718],
                [6.6462, 1.5423, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9.5141, 3.0926, 0.0],
                [2.9168, 0.0, 0.0, 5.5051, 9.279],
                [8.3815, 0.0, 0.0, 2.9814, 8.7799],
                [0.0, 0.0, 0.0, 0.0, 7.3127],
                [12.2062, 0.0, 9.0318, 3.2547, 0.0],
                [3.2526, 13.8423, 0.0, 14.0818, 0.0],
                [7.3971, 0.7834, 0.0, 0.8258, 0.0],
                [0.0, 3.214, 13.301, 0.0, 0.0],
                [0.0, 8.2386, 2.9588, 0.0, 4.6194],
                [0.8737, 8.7178, 0.0, 0.0, 0.0],
                [0.0, 2.6651, 2.3495, 0.046, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        super().__init__(
            name="Detergent optimization",
            inputs=[
                Continuous("x1", domain=[0.0, 0.2]),
                Continuous("x2", domain=[0.0, 0.3]),
                Continuous("x3", domain=[0.02, 0.2]),
                Continuous("x4", domain=[0.0, 0.06]),
                Continuous("x5", domain=[0.0, 0.04]),
            ],
            outputs=[Continuous(f"y{i+1}", domain=[0, 3]) for i in range(5)],
            objectives=[Maximize(f"y{i+1}") for i in range(5)],
            constraints=[
                LinearInequality(["x1", "x2", "x3", "x4", "x5"], lhs=-1, rhs=-0.2),
                LinearInequality(["x1", "x2", "x3", "x4", "x5"], lhs=1, rhs=0.4),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = np.atleast_2d(X[self.inputs.names])
        xp = np.stack([_poly2(xi) for xi in x], axis=0)
        return pd.DataFrame(xp @ self.coef, columns=self.outputs.names, index=X.index)


class Detergent_NChooseKConstraint(Problem):
    """Variant of the Detergent problem where only 3 of the 5 formulation components are allowed to be active (n-choose-k constraint)."""

    def __init__(self):
        base = Detergent()

        super().__init__(
            name="Detergent optimization with n-choose-k constraint",
            inputs=base.inputs,
            outputs=base.outputs,
            objectives=base.objectives,
            constraints=list(base.constraints)
            + [NChooseK(names=base.inputs.names, max_active=3)],
            f=base.f,
        )


class Detergent_OutputConstraint(Problem):
    """Variant of the Detergent problem with an additional output/black-box constraint.

    In addition to the 5 washing performances there is a sixth output reflecting the stability of the formulation.
    If `discrete=True` the stability can only be measured qualitatively (0: not stable, 1: stable).
    If `discrete=True` the stability can be measured quantively with smaller values indicating less stable formulations.
    """

    def __init__(self, discrete=False):
        base = Detergent()

        def f(X):
            Y = base.f(X)
            if discrete:
                Y["stable"] = (X.sum(axis=1) < 0.3).astype(int)
            else:
                Y["stable"] = (0.4 - X.sum(axis=1)) / 0.2
            return Y

        outputs = list(base.outputs)
        if discrete:
            outputs += [Discrete("stable", domain=[0, 1])]
        else:
            outputs += [Continuous("stable", domain=[0, 1])]

        super().__init__(
            name="Detergent optimization with stability constraint",
            inputs=base.inputs,
            outputs=outputs,
            objectives=base.objectives,
            output_constraints=[Maximize("stable", target=0.5)],
            constraints=base.constraints,
            f=f,
        )


class Detergent_TwoOutputConstraints(Problem):
    """Variant of the Detergent problem with two outputs constraint.

    In addition to the 5 washing performances there are two more outputs measuring the formulation stability.
    The first, stability 1, measures the immediate stability. If not stable, the other properties cannot be measured, except for stability 2.
    The second, stability 2, measures the long-term stability.
    """

    def __init__(self):
        base = Detergent()

        def f(X: pd.DataFrame) -> pd.DataFrame:
            Y = base.f(X)
            x = self.get_X(X)
            stable1 = (x.sum(axis=1) < 0.3).astype(int)
            stable2 = (x[:, :-1].sum(axis=1) < 0.25).astype(int)
            Y[stable1 == 0] = np.nan
            Y["stability 1"] = stable1
            Y["stability 2"] = stable2
            return Y

        outputs = list(base.outputs) + [
            Discrete("stability 1", domain=[0, 1]),
            Discrete("stability 2", domain=[0, 1]),
        ]

        super().__init__(
            name="Detergent optimization with two output constraint",
            inputs=base.inputs,
            outputs=outputs,
            objectives=base.objectives,
            output_constraints=[
                Maximize("stability 1", target=0.5),
                Maximize("stability 2", target=0.5),
            ],
            constraints=base.constraints,
            f=f,
        )
