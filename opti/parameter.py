import numbers
import pprint
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_CAT_SEP = "§"


class Parameter:
    """Parameter base class."""

    def __init__(self, name: str, domain: Sequence, type: str = None, **kwargs):
        self.name = name
        self.domain = domain
        self.type = type
        self.extra_fields = kwargs

    def to_config(self) -> Dict:
        """Return a json-serializable configuration dict."""
        conf = dict(name=self.name, type=self.type, domain=self.domain)
        conf.update(self.extra_fields)
        return conf


class Continuous(Parameter):
    """Parameter that can take on any real value in the specified domain.

    Attributes:
        name (str): name of the parameter
        domain (list): [lower bound, upper bound]
    """

    def __init__(
        self,
        name: str,
        domain: Optional[Sequence] = None,
        **kwargs,
    ):
        if domain is None:
            domain = [-np.inf, np.inf]
        else:
            if len(domain) != 2:
                raise ValueError("domain needs to consist of two values [low, high]")
        # convert None to inf (json doesn't support infinity)
        low = -np.inf if domain[0] is None else domain[0]
        high = np.inf if domain[1] is None else domain[1]
        if high < low:
            raise ValueError(f"lower bound {low} must be less than upper bound {high}")
        self.low = low
        self.high = high
        super().__init__(name=name, domain=[low, high], type="continuous", **kwargs)

    def __repr__(self):
        if np.isfinite(self.low) or np.isfinite(self.high):
            return f"Continuous(name='{self.name}', domain={self.domain})"
        else:
            return f"Continuous(name='{self.name}')"

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return the domain bounds."""
        return self.low, self.high

    def contains(self, point):
        """Check if a point is in contained in the domain.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point` with boolean datatype.
        """
        return (self.low <= point) & (point <= self.high)

    def round(self, point):
        """Round a point to the closest contained values.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point` with values clipped to parameter bounds.
        """
        return np.clip(point, self.low, self.high)

    def sample(self, n: int = 1) -> pd.Series:
        """Draw random samples from the domain."""
        low = max(self.low, np.finfo(np.float32).min)
        high = min(self.high, np.finfo(np.float32).max)
        return pd.Series(name=self.name, data=np.random.uniform(low, high, n))

    def to_config(self) -> Dict:
        """Return a json-serializable configuration dict."""
        low = None if np.isinf(self.low) else float(self.low)
        high = None if np.isinf(self.high) else float(self.high)
        conf = dict(name=self.name, type=self.type, domain=[low, high])
        conf.update(self.extra_fields)
        return conf

    def to_unit_range(self, points):
        """Transform points to the unit range: [low, high] -> [0, 1]."""
        if np.isclose(self.low, self.high):
            return points
        else:
            return (points - self.low) / (self.high - self.low)

    def from_unit_range(self, points):
        """Transform points from the unit range: [0-1] -> [low, high]."""
        if np.isclose(self.low, self.high):
            return np.ones_like(points) * self.high
        else:
            return points * (self.high - self.low) + self.low


class Discrete(Parameter):
    """Discrete parameter (ordinal scale).

    Attributes:
        name (str): name of the parameter
        domain (list): list of possible numeric values
    """

    def __init__(self, name: str, domain: Sequence, **kwargs):
        if len(domain) < 1:
            raise ValueError("domain must contain at least 1 value")
        for d in domain:
            if not isinstance(d, numbers.Number):
                raise ValueError("domain contains non-numeric values")
        if len(set(domain)) != len(domain):
            raise ValueError("domain contains duplicates")
        # convert to python-native dtype to ensure json-serializability
        domain = np.array(np.sort(domain)).tolist()
        self.low = min(domain)
        self.high = max(domain)
        super().__init__(name, domain, type="discrete", **kwargs)

    def __repr__(self):
        return f"Discrete(name='{self.name}', domain={self.domain})"

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return the domain bounds."""
        return self.low, self.high

    def is_integer(self):
        """Check if the domain is an integer range, such as [1, 2, 3]"""
        if isinstance(self.low, int) and isinstance(self.high, int):
            return self.domain == list(range(self.low, self.high + 1))
        else:
            return False

    def contains(self, point):
        """Check if a point is in contained in the domain.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point` with boolean datatype.
        """
        if not np.isscalar(point):
            point = np.array(point)
        return np.isin(point, self.domain)

    def round(self, point):
        """Round a point to the closest contained values.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point` with values clipped to parameter bounds.
        """
        if np.isscalar(point):
            i = np.argmin(np.abs(np.array(self.domain) - point))
            return self.domain[i]
        closest = [np.argmin(np.abs(np.array(self.domain) - p)) for p in point]
        rounded = np.array(self.domain)[closest]
        if isinstance(point, np.ndarray):
            return rounded
        elif isinstance(point, pd.Series):
            return pd.Series(name=self.name, data=rounded, index=point.index)
        elif isinstance(point, pd.DataFrame):
            return pd.DataFrame({self.name: rounded}, index=point.index)

    def sample(self, n: int = 1) -> pd.Series:
        """Draw random samples from the domain."""
        return pd.Series(name=self.name, data=np.random.choice(self.domain, n))

    def to_unit_range(self, points):
        """Transform points to the unit range.

        Note, if the given points are not inside the domain of the parameter, the
        transformed will not be inside the unit range.
        """
        if np.isclose(self.low, self.high):
            return points
        else:
            return (points - self.low) / (self.high - self.low)


class Categorical(Parameter):
    """Categorical parameter (nominal scale, values cannot be put into order).

    Attributes:
        name (str): name of the parameter
        domain (list): list possible values
    """

    def __init__(self, name: str, domain: Sequence, **kwargs):
        if not isinstance(domain, list):
            raise ValueError("domain must be of type list")
        if len(domain) < 2:
            raise ValueError("domain must a least contain 2 values")
        if len(set(domain)) != len(domain):
            raise ValueError("domain contains duplicates")
        super().__init__(name, domain, type="categorical", **kwargs)

    def __repr__(self):
        return f"Categorical(name='{self.name}', domain={self.domain})"

    def contains(self, point):
        """Check if a point is in contained in the domain.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point` with boolean datatype.
        """
        if not np.isscalar(point):
            point = np.array(point)
        return np.isin(point, self.domain)

    def round(self, point):
        """Round a point to the closest contained values.

        Args:
            point (float, np.ndarray, pd.Series or pd.Dataframe): parameter value(s).

        Returns:
            Object of the same type as `point`.
        """
        if not np.all(self.contains(point)):
            raise ValueError("cannot round categorical variable")
        return point

    def sample(self, n: int = 1) -> pd.Series:
        """Draw random samples from the domain."""
        return pd.Series(name=self.name, data=np.random.choice(self.domain, n))

    def to_onehot_encoding(self, points: pd.Series) -> pd.DataFrame:
        """Convert points to a one-hot encoding."""
        return pd.DataFrame(
            {f"{self.name}{_CAT_SEP}{c}": points == c for c in self.domain}, dtype=float
        )

    def from_onehot_encoding(self, points: pd.DataFrame) -> pd.Series:
        """Convert points brack from one-hot encoding."""
        cat_cols = [f"{self.name}{_CAT_SEP}{c}" for c in self.domain]
        if np.any([c not in cat_cols for c in points.columns]):
            raise ValueError(
                f"Column names don't match categorical levels: {points.columns}, {cat_cols}"
            )
        s = points.idxmax(1).str.split(_CAT_SEP, expand=True)[1]
        s.name = self.name
        return s

    def to_label_encoding(self, points: pd.Series) -> pd.Series:
        """Convert points to label-encoding."""
        enc = pd.Series(range(len(self.domain)), index=list(self.domain))
        s = enc[points]
        s.index = points.index
        s.name = self.name
        return s

    def from_label_encoding(self, points: pd.Series) -> pd.Series:
        """Convert points back from label-encoding."""
        enc = np.array(self.domain)
        return pd.Series(enc[points], index=points.index)


def make_parameter(
    name: str,
    type: str,
    domain: Optional[Sequence] = None,
    **kwargs,
):
    """Make a parameter object from a configuration

    p = make_parameter(**config)

    Args:
        type (str): "continuous", "discrete" or "categorical"
        name (str): Name of the parameter
        domain (list): Domain, e.g, [0, 1], [1, 2.5, 5] or ["A", "B", "C"]
    """
    parameter = {
        "continuous": Continuous,
        "discrete": Discrete,
        "categorical": Categorical,
    }[type.lower()]
    if domain is None and parameter is not Continuous:
        raise ValueError(f"domain not given for parameter {name}")

    return parameter(name=name, domain=domain, **kwargs)


class Parameters:
    """Set of parameters representing either the input or the output space."""

    def __init__(self, parameters):
        """
        It can be constructed either from a list / tuple (of at least one) Parameter objects
        ```
        Space([
            Continuous(name="foo", domain=[1, 10]),
            Discrete(name="bar", domain=[1, 2, 3, 4]),
            Categorical(name="baz", domain=["A", "B", 3]),
        ])
        ```
        or from a list / tuple of dicts
        ```
        Space([
            {"name": "foo", "type": "continuous", "domain": [1, 10]},
            {"name": "bar", "type": "discrete", "domain": [1, 2, 3, 4]},
            {"name": "baz", "type": "categorical", "domain": ["A", "B", 3]},
            {"name": "baz", "type": "categorical", "domain": ["A", "B", 3], extra="info"},
        ])
        ```
        In particular, Parameters().__init__ and the to_config method of each Parameter type are inverses.
        Parameters(conf).to_config() == conf
        """
        if not isinstance(parameters, (list, tuple)):
            raise TypeError("Space expects a list or tuple of parameters.")

        self.parameters = {}
        for d in parameters:
            if not isinstance(d, Parameter):
                d = make_parameter(**d)
            if d.name in self.parameters:
                raise ValueError(f"Duplicate parameter name {d.name}")
            self.parameters[d.name] = d

    def __repr__(self):
        return "Parameters(\n" + pprint.pformat(list(self.parameters.values())) + "\n)"

    def __iter__(self):
        return iter(self.parameters.values())

    def __getitem__(self, name):
        return self.parameters[name]

    def __len__(self):
        return len(self.parameters)

    def __add__(self, other):
        parameters = list(self.parameters.values()) + list(other.parameters.values())
        return Parameters(parameters)

    @property
    def names(self):
        return list(self.parameters.keys())

    @property
    def bounds(self) -> pd.DataFrame:
        """Return the parameter bounds."""
        for param in self:
            if isinstance(param, Categorical):
                raise TypeError(
                    "Contains categorical parameters which are not bounded."
                )
        return pd.DataFrame({p.name: p.bounds for p in self}, index=["min", "max"])

    def contains(self, points: pd.DataFrame) -> pd.Series:
        """Check if points are inside the space in each parameter."""
        b = np.stack([self[k].contains(v) for k, v in points.iteritems()], axis=-1)
        return b.all(axis=-1)

    def round(self, points: pd.DataFrame) -> pd.DataFrame:
        """Round points to the closest contained values."""
        return pd.concat([self[k].round(v) for k, v in points.iteritems()], axis=1)

    def sample(self, n: int = 1) -> pd.DataFrame:
        """Draw uniformly distributed random samples."""
        return pd.concat([param.sample(n) for param in self], axis=1)

    def transform(
        self,
        points: pd.DataFrame,
        continuous: str = "none",
        discrete: str = "none",
        categorical: str = "onehot-encode",
    ) -> pd.DataFrame:
        transformed = []
        for p in self:
            s = points[p.name]
            if isinstance(p, Continuous):
                if continuous == "none":
                    transformed.append(s)
                elif continuous == "normalize":
                    transformed.append(p.to_unit_range(s))
                else:
                    ValueError(f"Unknown continuous transform {continuous}")
            if isinstance(p, Discrete):
                if discrete == "none":
                    transformed.append(s)
                elif discrete == "normalize":
                    transformed.append(p.to_unit_range(s))
                else:
                    ValueError(f"Unknown discrete transform {continuous}")
            if isinstance(p, Categorical):
                if categorical == "none":
                    transformed.append(s)
                elif categorical == "onehot-encode":
                    transformed.append(p.to_onehot_encoding(s))
                elif categorical == "label-encode":
                    transformed.append(p.to_label_encoding(s))
                else:
                    ValueError(f"Unknown categorical transform {continuous}")
        return pd.concat(transformed, axis=1)

    def to_config(self) -> List[Dict]:
        """Configuration of the parameter space."""
        return [param.to_config() for param in self.parameters.values()]
