import pprint
from typing import List, Optional, Sequence, Tuple

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

    def to_config(self) -> dict:
        """Return a json-serializable configuration dict."""
        conf = dict(name=self.name, type=self.type, domain=self.domain)
        conf.update(self.extra_fields)
        return conf


class Continuous(Parameter):
    """Variable that can take on any real value in the specified domain.

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
                raise ValueError(
                    f"{name}: Domain must consist of two values [low, high]."
                )
        # convert None to +/- inf and string to float
        low = -np.inf if domain[0] is None else float(domain[0])
        high = np.inf if domain[1] is None else float(domain[1])

        if high < low:
            raise ValueError(
                f"{name}: Lower bound {low} must be less than upper bound {high}."
            )
        self.low = low
        self.high = high
        super().__init__(name=name, domain=[low, high], type="continuous", **kwargs)

    def __repr__(self):
        if np.isfinite(self.low) or np.isfinite(self.high):
            return f"Continuous('{self.name}', domain={self.domain})"
        else:
            return f"Continuous('{self.name}')"

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

    def to_config(self) -> dict:
        """Return a json-serializable configuration dict."""
        conf = dict(name=self.name, type=self.type)
        low = None if np.isinf(self.low) else float(self.low)
        high = None if np.isinf(self.high) else float(self.high)
        if low is not None or high is not None:
            conf.update({"domain": [low, high]})
        conf.update(self.extra_fields)
        return conf

    def to_unit_range(self, points):
        """Transform points to the unit range: [low, high] -> [0, 1].

        Points outside of the domain will transform to outside of [0, 1].
        Nothing is done if low == high.
        """
        if np.isclose(self.low, self.high):
            return points
        else:
            return (points - self.low) / (self.high - self.low)

    def from_unit_range(self, points):
        """Transform points from the unit range: [0, 1] -> [low, high].

        A rounding is applied to correct for numerical precision.
        Nothing is done if low == high.
        """
        if np.isclose(self.low, self.high):
            return points
        else:
            return points * (self.high - self.low) + self.low


class Discrete(Parameter):
    """Variable with a discrete domain (ordinal scale).

    Attributes:
        name (str): name of the parameter
        domain (list): list of possible numeric values
    """

    def __init__(self, name: str, domain: Sequence, **kwargs):
        if len(domain) < 1:
            raise ValueError(f"{name}: Domain must contain at least one value.")
        try:
            # convert to a sorted list of floats
            domain = np.sort(np.array(domain).astype(float)).tolist()
        except ValueError:
            raise ValueError(f"{name}: Domain contains non-numeric values.")
        if len(set(domain)) != len(domain):
            raise ValueError(f"{name}: Domain contains duplicates.")
        self.low = min(domain)
        self.high = max(domain)
        super().__init__(name, domain, type="discrete", **kwargs)

    def __repr__(self):
        return f"Discrete('{self.name}', domain={self.domain})"

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
        """Transform points to the unit range: [low, high] -> [0, 1].

        Points outside of the domain will transform to outside of [0, 1].
        Nothing is done if low == high.
        """
        if np.isclose(self.low, self.high):
            return points
        else:
            return (points - self.low) / (self.high - self.low)

    def from_unit_range(self, points):
        """Transform points from the unit range: [0, 1] -> [low, high].

        A rounding is applied to correct for numerical precision.
        Nothing is done if low == high.
        """
        if np.isclose(self.low, self.high):
            return points
        else:
            points = points * (self.high - self.low) + self.low
            return self.round(points)


class Categorical(Parameter):
    """Variable with a categorical domain (nominal scale, values cannot be put into order).

    Attributes:
        name (str): name of the parameter
        domain (list): list possible values
    """

    def __init__(self, name: str, domain: List[str], **kwargs):
        if not isinstance(domain, list):
            raise TypeError(f"{name}: Domain must be of type list.")
        if len(domain) < 2:
            raise ValueError(f"{name}: Domain must a least contain 2 values.")
        if len(set(domain)) != len(domain):
            raise ValueError(f"{name}: Domain contains duplicates.")
        super().__init__(name, domain, type="categorical", **kwargs)

    def __repr__(self):
        return f"Categorical('{self.name}', domain={self.domain})"

    @property
    def bounds(self) -> Tuple[float, float]:
        """Return the domain bounds."""
        return np.nan, np.nan

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
            raise ValueError(
                f"{self.name}: Cannot round values for categorical parameter."
            )
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
        """Convert points back from one-hot encoding."""
        cat_cols = [f"{self.name}{_CAT_SEP}{c}" for c in self.domain]
        if np.any([c not in cat_cols for c in points.columns]):
            raise ValueError(
                f"{self.name}: Column names don't match categorical levels: {points.columns}, {cat_cols}."
            )
        s = points.idxmax(1).str.split(_CAT_SEP, expand=True)[1]
        s.name = self.name
        return s

    def to_dummy_encoding(self, points: pd.Series) -> pd.DataFrame:
        """Convert points to a dummy-hot encoding, dropping the first categorical level."""
        return pd.DataFrame(
            {f"{self.name}{_CAT_SEP}{c}": points == c for c in self.domain[1:]},
            dtype=float,
        )

    def from_dummy_encoding(self, points: pd.DataFrame) -> pd.Series:
        """Convert points back from dummy encoding."""
        cat_cols = [f"{self.name}{_CAT_SEP}{c}" for c in self.domain]
        if np.any([c not in cat_cols[1:] for c in points.columns]):
            raise ValueError(
                f"{self.name}: Column names don't match categorical levels: {points.columns}, {cat_cols}."
            )
        points = points.copy()
        points[cat_cols[0]] = 1 - points[cat_cols[1:]].sum(axis=1)
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
        raise ValueError(f"Domain not given for parameter {name}.")

    return parameter(name=name, domain=domain, **kwargs)


class Parameters:
    """Set of parameters representing either the input or the output parameter space."""

    def __init__(self, parameters):
        """
        It can be constructed either from a list / tuple (of at least one) Parameter objects
        ```
        Parameters([
            Continuous(name="foo", domain=[1, 10]),
            Discrete(name="bar", domain=[1, 2, 3, 4]),
            Categorical(name="baz", domain=["A", "B", 3]),
        ])
        ```
        or from a list / tuple of dicts
        ```
        Parameters([
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
            raise TypeError("Parameters expects a list or tuple of parameters.")

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
        return pd.DataFrame({p.name: p.bounds for p in self}, index=["min", "max"])

    def contains(self, points: pd.DataFrame) -> pd.Series:
        """Check if points are inside the domain of each parameter."""
        if isinstance(points, pd.DataFrame):
            points = points[self.names]
        b = np.stack([self[k].contains(v) for k, v in points.iteritems()], axis=1)
        return b.all(axis=1)

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
        """Transfrom the given dataframe according to a set of transformation rules.

        Args:
            points (pd.DataFrame): Dataframe to transfrom. Must contain columns for each parameter and may contain additional columns.
            continuous (str, optional): Transform for continuous parameters. Options are
                - "none" (default): keep values unchanged.
                - "normalize": transforms the domain bounds to [0, 1]
            discrete (str, optional): Transform for discrete parameters. Options are
                - "none" (default): keep values unchanged.
                - "normalize": transforms the domain bounds to [0, 1]
            categorical (str, optional): Transform for categoricals. Options are
                - "onehot-encode" (default): A parameter with levels [A, B, C] transforms to 3 columns holding values [0, 1]
                - "dummy-encode": a parameter with levels [A, B, C] transforms to 2 columns holding values [0, 1]
                - "label-encode": a parameter with levels [A, B, C] transfroms to 1 columns with values [0, 1, 2]
                - "none": keep values unchanged

        Raises:
            ValueError: Unknown transform.

        Returns:
            pd.DataFrame: Transformed points. Columns that don't correspond to parameters are dropped.
        """
        transformed = []
        for p in self:
            s = points[p.name]
            if isinstance(p, Continuous):
                if continuous == "none":
                    transformed.append(s)
                elif continuous == "normalize":
                    transformed.append(p.to_unit_range(s))
                else:
                    raise ValueError(f"Unknown continuous transform {continuous}.")
            if isinstance(p, Discrete):
                if discrete == "none":
                    transformed.append(s)
                elif discrete == "normalize":
                    transformed.append(p.to_unit_range(s))
                else:
                    raise ValueError(f"Unknown discrete transform {discrete}.")
            if isinstance(p, Categorical):
                if categorical == "none":
                    transformed.append(s)
                elif categorical == "onehot-encode":
                    transformed.append(p.to_onehot_encoding(s))
                elif categorical == "dummy-encode":
                    transformed.append(p.to_dummy_encoding(s))
                elif categorical == "label-encode":
                    transformed.append(p.to_label_encoding(s))
                else:
                    raise ValueError(f"Unknown categorical transform {categorical}.")
        return pd.concat(transformed, axis=1)

    def to_config(self) -> List[dict]:
        """Configuration of the parameter space."""
        return [param.to_config() for param in self.parameters.values()]

    def get(self, types) -> "Parameters":
        """Get all parameters of the given type(s)."""
        return Parameters([p for p in self if isinstance(p, types)])

    def to_df(self, x: np.ndarray, to_numeric=False) -> pd.DataFrame:
        """Create a dataframe for a given numpy array of parameter values."""
        X = pd.DataFrame(np.atleast_2d(x), columns=self.names)
        if to_numeric:
            for n in self.get((Continuous, Discrete)).names:
                X[n] = pd.to_numeric(X[n])
        return X
