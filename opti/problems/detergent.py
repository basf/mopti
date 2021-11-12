import numpy as np
import pandas as pd

from opti.constraint import LinearInequality, NChooseK
from opti.objective import Maximize
from opti.parameter import Continuous
from opti.problem import Problem


def poly2(x: np.ndarray) -> np.ndarray:
    """Full quadratic feature expansion including bias term."""
    return np.concatenate([[1], x, np.outer(x, x)[np.triu_indices(5)]])


class Detergent(Problem):
    """Constrained problem with 5 inputs and 5 outputs.

    Each output is modeled as a second degree polynomial.

    The sixth input is a filler (water) and is factored out using the formulation
    constraint sum x = 1, and it's parameter bounds 0.6 < water < 0.8 result in 2 linear
    inequality constraints for the other parameters.
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
            name="Detergent",
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
                LinearInequality(
                    names=["x1", "x2", "x3", "x4", "x5"], lhs=-1, rhs=-0.2
                ),
                LinearInequality(names=["x1", "x2", "x3", "x4", "x5"], lhs=1, rhs=0.4),
            ],
        )

    def f(self, df: pd.DataFrame) -> pd.DataFrame:
        x = np.atleast_2d(df[self.inputs.names])
        xp = np.stack([poly2(xi) for xi in x], axis=0)
        return pd.DataFrame(xp @ self.coef, columns=self.outputs.names)


class Detergent_NChooseKConstraint(Problem):
    """Variant of the Detergent problem with an n-choose-k constraint"""

    def __init__(self):
        base = Detergent()

        super().__init__(
            name="Detergent with n-choose-k constraint",
            inputs=base.inputs,
            outputs=base.outputs,
            objectives=base.objectives,
            constraints=list(base.constraints)
            + [NChooseK(names=base.inputs.names, max_active=3)],
            f=base.f,
        )


class Detergent_OutputConstraint(Problem):
    """Variant of the Detergent problem with an output constraint.

    There are 5 inputs, 6 outputs:
    - 3 outputs are to be maximized
    - 1 output represents the stability of the formulation
        (0: not stable, 1: stable)
    """

    def __init__(self):
        base = Detergent()

        def f(df):
            Y = base.f(df)
            Y["stable"] = (0.4 - df.sum(axis=1)) / 0.2  # continuous version
            Y["stable"] = df.sum(axis=1) < 0.3  # discrete version
            return Y

        super().__init__(
            name="Detergent with stability constraint",
            inputs=base.inputs,
            outputs=list(base.outputs) + [Continuous("stable", domain=[0, 1])],
            objectives=[Maximize(f"y{i+1}") for i in range(3)],
            output_constraints=[Maximize("stable", target=0.5)],
            constraints=base.constraints,
            f=f,
        )
