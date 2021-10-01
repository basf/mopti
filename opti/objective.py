import pprint
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from opti.parameter import Parameters


class Objective:
    def __init__(self, name: str, parameter: Union[str, List[str]]):
        """Base class for optimzation objectives.

        Args:
            name: name of the objective
            parameters: parameter(s) that the objective is operating on
        """
        self.name = name
        self.parameter = parameter

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Evaluate the objective values for given output values."""
        raise NotImplementedError  # implemented in the derived classes

    def eval(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate the objective values for a given DataFrame."""
        return pd.Series(
            self(df[self.parameter].values), index=df.index, name=self.name
        )

    def to_config(self) -> Dict:
        """Return a json-serializable dictionary of the objective."""
        raise NotImplementedError  # implemented in the derived classes


class Minimize(Objective):
    def __init__(self, parameter: str, target: float = 0):
        """Minimization objective

        s(y) = y - target

        Args:
            parameter: parameter to minimize
            target: value below which no further improvement is required
        """
        super().__init__(name=parameter, parameter=parameter)
        self.target = target

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return y - self.target

    def untransform(self, y: np.ndarray) -> np.ndarray:
        """Undo the transformation from output to objective value"""
        return y + self.target

    def __repr__(self):
        return f"Minimize('{self.parameter}', target={self.target})"

    def to_config(self) -> Dict:
        config = dict(name=self.parameter, type="minimize")
        if self.target != 0:
            config["target"] = self.target
        return config


class Maximize(Objective):
    def __init__(self, parameter: str, target: float = 0):
        """Maximization objective

        s(y) = target - y

        Args:
            name: name of the objective (= parameter to optimize)
            target: value above which no further improvement is required
        """
        super().__init__(name=parameter, parameter=parameter)
        self.target = target

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return -y + self.target

    def untransform(self, y: np.ndarray) -> np.ndarray:
        """Undo the transformation from output to objective value"""
        return -y - self.target

    def __repr__(self):
        return f"Maximize('{self.parameter}', target={self.target})"

    def to_config(self) -> Dict:
        config = dict(name=self.parameter, type="maximize")
        if self.target != 0:
            config["target"] = str(self.target)
        return config


class CloseToTarget(Objective):
    def __init__(
        self,
        parameter: str,
        target: float = 0,
        exponent: float = 1,
        tolerance: float = 0,
    ):
        """Objective for getting as close as possible to a given value.

        s(y) = |y - target| ** exponent - tolerance ** exponent

        Args:
            parameter: parameter to optimize
            target: target value
            exponent: exponent of the difference
            tolerance: distance to target below which no further improvement is required
        """
        super().__init__(name=parameter, parameter=parameter)
        self.target = target
        self.exponent = exponent
        self.tolerance = tolerance

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return (
            np.abs(y - self.target) ** self.exponent - self.tolerance ** self.exponent
        )

    def __repr__(self):
        return f"CloseToTarget('{self.parameter}', target={self.target})"

    def to_config(self) -> Dict:
        config = dict(name=self.parameter, type="close-to-target", target=self.target)
        if self.exponent != 1:
            config["exponent"] = self.exponent
        if self.tolerance != 0:
            config["tolerance"] = self.tolerance
        return config


class Objectives:
    """Container for optimization objectives.

    Objectives can be either used to quantify the optimility or as a constraint on the
    viability of output values (chance / feasibility constraint)
    """

    def __init__(self, objectives: Union[List[Objective], List[Dict]]):
        _objectives = []
        for m in objectives:
            if isinstance(m, Objective):
                _objectives.append(m)
            else:
                _objectives.append(make_objective(**m))
        self.objectives = _objectives

    def __call__(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(y)
        return np.stack([obj(yi) for yi, obj in zip(y.T, self.objectives)], axis=-1)

    def eval(self, y: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([obj.eval(y) for obj in self.objectives], axis=1)

    def __repr__(self):
        return "Objectives(\n" + pprint.pformat(self.objectives) + "\n)"

    def __iter__(self):
        return iter(self.objectives)

    def __len__(self):
        return len(self.objectives)

    def __getitem__(self, i: int) -> Objective:
        return self.objectives[i]

    @property
    def names(self):
        return [obj.name for obj in self]

    def bounds(self, outputs: Parameters) -> pd.DataFrame:
        """Compute the bounds in objective space based on the output space bounds.

        The bounds can be interpreted as the ideal and nadir points.

        Examples for continuous parameters:
            min y for y in [0, 10] -> ideal = 0, nadir = 10
            max y for y in [0, 10] -> ideal = -10, nadir = 0
            min (y - 7)**2 for y in [0, 10] -> ideal = 0, nadir = 7**2

        Args:
            outputs: Output parameters.
        """
        Z = self.eval(outputs.bounds)
        bounds = pd.DataFrame(columns=self.names, dtype=float)
        bounds.loc["min"] = Z.min(axis=0)
        bounds.loc["max"] = Z.max(axis=0)

        for name, obj in zip(self.names, self):
            if isinstance(obj, CloseToTarget):
                bounds.loc["min", name] = 0

        return bounds

    def to_config(self) -> List[Dict]:
        return [obj.to_config() for obj in self.objectives]


def make_objective(type: str, name: Union[str, List[str]], **kwargs) -> Objective:
    """Make an objective from a configuration.

    ```
    obj = make_objective(**config)
    ```

    Args:
        type: objective type
        name: parameter(s) that the objective is operating on
    """
    objective = {
        "minimize": Minimize,
        "maximize": Maximize,
        "close-to-target": CloseToTarget,
    }[type.lower()]
    return objective(parameter=name, **kwargs)
