import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from opti.constraint import Constraints, LinearEquality, NChooseK, NonlinearEquality
from opti.parameter import Continuous, Parameters
from opti.sampling.polytope import polytope_sampling


def split_nchoosek(
    constraints: Optional[Constraints],
) -> Tuple[Constraints, Constraints]:
    """Split constraints in n-choose-k constraint and all other constraints."""
    if constraints is None:
        return Constraints([]), Constraints([])
    nchoosek_constraints = Constraints(
        [c for c in constraints if isinstance(c, NChooseK)]
    )
    other_constraints = Constraints(
        [c for c in constraints if not isinstance(c, NChooseK)]
    )
    return nchoosek_constraints, other_constraints


def apply_nchoosek(samples: pd.DataFrame, constraint: NChooseK):
    """Apply an n-choose-k constraint in-place"""
    n_zeros = len(constraint.names) - constraint.max_active
    for i in samples.index:
        s = np.random.choice(constraint.names, size=n_zeros, replace=False)
        samples.loc[i, s] = 0


def constrained_sampling(
    n_samples: int, parameters: Parameters, constraints: Constraints
) -> pd.DataFrame:
    """Uniform sampling from a constrained space."""
    nchoosek_constraints, other_constraints = split_nchoosek(constraints)

    try:
        samples = rejection_sampling(n_samples, parameters, other_constraints)
    except Exception:
        samples = polytope_sampling(n_samples, parameters, other_constraints)

    if len(nchoosek_constraints) == 0:
        return samples

    for c in nchoosek_constraints:
        apply_nchoosek(samples, c)
    # check if other constraints are still satisfied
    if not constraints.satisfied(samples).all():
        raise Exception(
            "Applying the n-choose-k constraint(s) violated another constraint."
        )
    return samples


def sobol_sampling(n_samples: int, parameters: Parameters) -> pd.DataFrame:
    """Super-uniform sampling from an unconstrained space using a Sobol sequence."""
    d = len(parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = Sobol(d).random(n_samples)
    res = []
    for i, p in enumerate(parameters):
        if isinstance(p, Continuous):
            x = p.from_unit_range(X[:, i])
        else:
            bins = np.linspace(0, 1, len(p.domain) + 1)
            idx = np.digitize(X[:, i], bins) - 1
            x = np.array(p.domain)[idx]
        res.append(pd.Series(x, name=p.name))
    return pd.concat(res, axis=1)


def rejection_sampling(
    n_samples: int,
    parameters: Parameters,
    constraints: Constraints,
    max_iters: int = 1000,
) -> pd.DataFrame:
    """Uniformly distributed samples from a constrained space via rejection sampling."""
    if constraints is None:
        return parameters.sample(n_samples)

    # check for equality constraints in combination with continuous parameters
    for c in constraints:
        if isinstance(c, (LinearEquality, NonlinearEquality)):
            for p in parameters:
                if isinstance(p, Continuous):
                    raise Exception(
                        "Rejection sampling doesn't work for equality constraints over continuous variables."
                    )

    n_iters = 0
    n_found = 0
    points_found = []
    while n_found < n_samples:
        n_iters += 1
        if n_iters > max_iters:
            raise Exception("Maximum iterations exceeded in rejection sampling")
        points = parameters.sample(10000)
        valid = constraints.satisfied(points)
        n_found += np.sum(valid)
        points_found.append(points[valid])

    return pd.concat(points_found, ignore_index=True).iloc[:n_samples]
