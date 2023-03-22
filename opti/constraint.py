import pprint
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class Constraint:
    """Base class to define constraints on the input space, g(x) == 0 or g(x) <= 0."""

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """Numerically evaluate the constraint g(x)."""
        raise NotImplementedError

    def jacobian(self, data: pd.DataFrame) -> pd.DataFrame:
        """Numerically evaluate the jacobian of the constraint J_g(x)"""
        raise NotImplementedError

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        """Check if a constraint is satisfied, i.e. g(x) == 0 for equalities and g(x) <= for inequalities."""
        raise NotImplementedError

    def to_config(self) -> Dict:
        raise NotImplementedError


class LinearEquality(Constraint):
    def __init__(
        self,
        names: List[str],
        lhs: Union[float, List[float], np.ndarray] = 1,
        rhs: float = 0,
    ):
        """Linear / affine inequality of the form 'lhs * x == rhs'.

        Args:
            names: Parameter names that the constraint works on.
            lhs: Left-hand side / coefficients of the constraint.
            rhs: Right-hand side of the constraint.

        Examples:
            A mixture constraint where A, B and C need to add up to 100 can be defined as
            ```
            LinearEquality(["A", "B", "C"], rhs=100)
            ```
            If the coefficients of A, B and C are not 1 they are passed explicitly.
            ```
            LinearEquality(["A", "B", "C"], lhs=[10, 2, 5], rhs=100)
            ```
        """
        self.names = names
        if np.isscalar(lhs):
            self.lhs = lhs * np.ones(len(names))
        else:
            self.lhs = np.asarray(lhs)
        if self.lhs.shape != (len(names),):
            raise ValueError("Number of parameters and coefficients/lhs don't match.")
        self.rhs = rhs
        self.is_equality = True

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return (data[self.names] @ self.lhs - self.rhs) / np.linalg.norm(self.lhs)

    def jacobian(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(self.lhs / np.linalg.norm(self.lhs), [data.shape[0], 1]),
            columns=["dg/d" + name for name in self.names],
        )

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(data), 0), index=data.index)

    def __repr__(self):
        return (
            f"LinearEquality(names={self.names}, lhs={list(self.lhs)}, rhs={self.rhs})"
        )

    def to_config(self) -> Dict:
        return dict(
            type="linear-equality",
            names=self.names,
            lhs=self.lhs.tolist(),
            rhs=self.rhs,
        )


class LinearInequality(Constraint):
    def __init__(
        self,
        names: List[str],
        lhs: Union[float, List[float], np.ndarray] = 1,
        rhs: float = 0,
    ):
        """Linear / affine inequality of the form 'lhs * x <= rhs'.

        Args:
            names: Parameter names that the constraint works on.
            lhs: Left-hand side / coefficients of the constraint.
            rhs: Right-hand side of the constraint.

        Examples:
            A mixture constraint where the values of A, B and C may not exceed 100 can be defined as
            ```
            LinearInequality(["A", "B", "C"], rhs=100)
            ```
            If the coefficients are not 1, they need to be passed explicitly.
            ```
            LinearInequality(["A", "B", "C"], lhs=[10, 2, 5], rhs=100)
            ```
            Inequalities are alway of the form g(x) <= 0. To define a the constraint g(x) >=0 0, both `lhs` and `rhs` need to be multiplied by -1.
            ```
            LinearInequality(["A", "B", "C"], lhs=-1, rhs=-100)
            LinearInequality(["A", "B", "C"], lhs=[-10, -2, -5], rhs=-100)
            ```
        """
        self.names = names
        if np.isscalar(lhs):
            self.lhs = lhs * np.ones(len(names))
        else:
            self.lhs = np.asarray(lhs)
        if self.lhs.shape != (len(names),):
            raise ValueError("Number of parameters and coefficients/lhs don't match.")
        self.rhs = rhs
        self.is_equality = False

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return (data[self.names] @ self.lhs - self.rhs) / np.linalg.norm(self.lhs)

    def jacobian(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(self.lhs / np.linalg.norm(self.lhs), [data.shape[0], 1]),
            columns=["dg/d" + name for name in self.names],
        )

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        return self(data) <= 0

    def __repr__(self):
        return f"LinearInequality(names={self.names}, lhs={list(self.lhs)}, rhs={self.rhs})"

    def to_config(self) -> Dict:
        return dict(
            type="linear-inequality",
            names=self.names,
            lhs=self.lhs.tolist(),
            rhs=self.rhs,
        )


class NonlinearEquality(Constraint):
    def __init__(
        self,
        expression: str,
        jacobian: Optional[str] = None,
        names: Optional[List[str]] = None,
    ):
        """Equality of the form 'expression == 0'.

        Args:
            expression: Mathematical expression that can be evaluated by `pandas.eval`.
            jacobian: List of mathematical expressions that can be evaluated by `pandas.eval`.
                The i-th expression should correspond to the partial derivative with respect to
                the i-th variable. If `names` attribute is provided, the order of the variables should
                correspond to the order of the variables in `names`. Optional.
            names: List of variable names present in `expression`. Optional.

        Examples:
            You can pass any expression that can be evaluated by `pd.eval`.
            To define x1**2 + x2**2 = 1, use
            ```
            NonlinearEquality("x1**2 + x2**2 - 1")
            ```
            Standard mathematical operators are supported.
            ```
            NonlinearEquality("sin(A) / (exp(B) - 1)")
            ```
            Parameter names with special characters or spaces need to be enclosed in backticks.
            ```
            NonlinearEquality("1 - `weight A` / `weight B`")
            ```
        """
        self.expression = expression
        self.is_equality = True
        self.jacobian_expression = jacobian
        self.names = names

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return data.eval(self.expression)

    def jacobian(self, data: pd.DataFrame) -> pd.DataFrame:

        if self.jacobian_expression is not None:
            res = data.eval(self.jacobian_expression)
            for i, col in enumerate(res):
                if not hasattr(col, "__iter__"):
                    res[i] = pd.Series(np.repeat(col, data.shape[0]))

            if self.names is not None:
                return pd.DataFrame(
                    res, index=["dg/d" + name for name in self.names]
                ).transpose()
            else:
                return pd.DataFrame(
                    res, index=[f"dg/dx{i}" for i in range(data.shape[1])]
                ).transpose()

        return super().jacobian(data)

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(data), 0), index=data.index)

    def __repr__(self):
        return f"NonlinearEquality('{self.expression}')"

    def to_config(self) -> Dict:
        return dict(type="nonlinear-equality", expression=self.expression)


class NonlinearInequality(Constraint):
    def __init__(
        self,
        expression: str,
        jacobian: Optional[str] = None,
        names: Optional[List[str]] = None,
    ):
        """Inequality of the form 'expression <= 0'.

        Args:
            expression: Mathematical expression that can be evaluated by `pandas.eval`.
            jacobian: List of mathematical expressions that can be evaluated by `pandas.eval`.
                The i-th expression should correspond to the partial derivative with respect to
                the i-th variable. If `names` attribute is provided, the order of the variables should
                correspond to the order of the variables in `names`. Optional.
            names: List of variable names present in `expression`. Optional.

        Examples:
            You can pass any expression that can be evaluated by `pd.eval`.
            To define x1**2 + x2**2 < 1, use
            ```
            NonlinearInequality("x1**2 + x2**2 - 1")
            ```
            Standard mathematical operators are supported.
            ```
            NonlinearInequality("sin(A) / (exp(B) - 1)")
            ```
            Parameter names with special characters or spaces need to be enclosed in backticks.
            ```
            NonlinearInequality("1 - `weight A` / `weight B`")
            ```
        """
        self.expression = expression
        self.is_equality = False
        self.jacobian_expression = jacobian
        self.names = names

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return data.eval(self.expression)

    def jacobian(self, data: pd.DataFrame) -> pd.DataFrame:

        if self.jacobian_expression is not None:
            res = data.eval(self.jacobian_expression)
            for i, col in enumerate(res):
                if not hasattr(col, "__iter__"):
                    res[i] = pd.Series(np.repeat(col, data.shape[0]))

            if self.names is not None:
                return pd.DataFrame(
                    res, index=["dg/d" + name for name in self.names]
                ).transpose()
            else:
                return pd.DataFrame(
                    res, index=[f"dg/dx{i}" for i in range(data.shape[1])]
                ).transpose()

        return super().jacobian(data)

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        return self(data) <= 0

    def __repr__(self):
        return f"NonlinearInequality('{self.expression}')"

    def to_config(self) -> Dict:
        return dict(type="nonlinear-inequality", expression=self.expression)


class NChooseK(Constraint):
    def __init__(self, names: List[str], max_active: int):
        """Only k out of n values are allowed to take nonzero values.

        Args:
            names: Parameter names that the constraint works on.
            max_active: Maximium number of non-zero parameter values.

        Examples:
            A choice of 2 or less from A, B, C, D or E can be defined as
            ```
            NChooseK(["A", "B", "C", "D", "E"], max_active=2)
            ```
        """
        self.names = names
        self.max_active = max_active
        self.is_equality = False

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        x = np.abs(data[self.names].values)
        num_zeros = x.shape[1] - self.max_active
        violation = np.apply_along_axis(
            func1d=lambda r: sum(sorted(r)[:num_zeros]), axis=1, arr=x
        )
        return pd.Series(violation, index=data.index)

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self(data) <= 0, index=data.index)

    def __repr__(self):
        return f"NChooseK(names={self.names}, max_active={self.max_active})"

    def to_config(self) -> Dict:
        return dict(type="n-choose-k", names=self.names, max_active=self.max_active)


class Constraints:
    """List of input constraints"""

    def __init__(self, constraints: Sequence):
        self.constraints = []
        for c in constraints:
            if not isinstance(c, Constraint):
                if "names" in c and len(c["names"]) == 0:
                    continue  # skip empty constraints
                c = make_constraint(**c)
            self.constraints.append(c)

    def __repr__(self):
        return "Constraints(\n" + pprint.pformat(self.constraints) + "\n)"

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, i):
        return self.constraints[i]

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Numerically evaluate all constraints.

        Args:
            data: Data to evaluate the constraints on.

        Returns:
            Constraint evaluation g(x) for each of the constraints.
        """
        return pd.concat([c(data) for c in self.constraints], axis=1)

    def jacobian(self, data: pd.DataFrame) -> List:
        """Numerically evaluate all constraint gradients.

        Args:
            data: Data to evaluate the constraint gradients on.

        Returns:
            Jacobian evaluation J_g(x) for each of the constraints as a list of dataframes.
        """
        return [c.jacobian(data) for c in self.constraints]

    def satisfied(self, data: pd.DataFrame) -> pd.Series:
        """Check if all constraints are satisfied.

        Args:
            data: Data to evaluate the constraints on.

        Returns:
            Series of booleans indicating if all constraints are satisfied.
        """
        return pd.concat([c.satisfied(data) for c in self.constraints], axis=1).all(
            axis=1
        )

    def to_config(self) -> List[Dict]:
        return [obj.to_config() for obj in self.constraints]

    def get(self, types) -> "Constraints":
        """Get all constraints of the given type(s)."""
        return Constraints([c for c in self if isinstance(c, types)])


def make_constraint(type, **kwargs):
    t = type.lower()
    if t == "linear-equality":
        return LinearEquality(**kwargs)
    if t == "linear-inequality":
        return LinearInequality(**kwargs)
    if t == "nonlinear-equality":
        return NonlinearEquality(**kwargs)
    if t == "nonlinear-inequality":
        return NonlinearInequality(**kwargs)
    if t == "n-choose-k":
        return NChooseK(**kwargs)
    raise ValueError(f"Unknown constraint type: {t}.")
