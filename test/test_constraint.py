import json

import numpy as np
import pandas as pd
import pytest

from opti.constraint import (
    Constraints,
    LinearEquality,
    LinearInequality,
    NChooseK,
    NonlinearEquality,
    NonlinearInequality,
    make_constraint,
)


def test_linear_equality():
    names = [f"x{i}" for i in range(5)]
    constraint = LinearEquality(names=names, lhs=np.ones(5), rhs=5)

    df = pd.DataFrame([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]], columns=names)
    assert np.allclose(constraint.eval(df), [0, 10])
    assert np.allclose(constraint.satisfied(df), [True, False])

    eval(constraint.__repr__())
    json.dumps(constraint.to_config())
    constraint2 = make_constraint(**constraint.to_config())
    assert np.allclose(constraint2.eval(df), constraint.eval(df))

    constraint = LinearEquality(names=names)
    assert np.allclose(constraint.lhs, 1)
    assert constraint.rhs == 0

    with pytest.raises(ValueError):
        LinearEquality(names=names, lhs=[1, 1, 1])


def test_linear_inequality():
    names = [f"x{i}" for i in range(5)]
    constraint = LinearInequality(names=names, lhs=np.ones(5), rhs=5)

    df = pd.DataFrame([[1, 0.1, 1, 1, 1], [1, 2, 3, 4, 5]], columns=names)
    assert np.allclose(constraint.eval(df), [-0.9, 10])
    assert np.allclose(constraint.satisfied(df), [True, False])

    eval(constraint.__repr__())
    json.dumps(constraint.to_config())
    constraint2 = make_constraint(**constraint.to_config())
    assert np.allclose(constraint2.eval(df), constraint.eval(df))

    constraint = LinearInequality(names=names)
    assert np.allclose(constraint.lhs, 1)
    assert constraint.rhs == 0

    with pytest.raises(ValueError):
        LinearInequality(names=names, lhs=[1, 1, 1])


def test_nonlinear_equality():
    constraint = NonlinearEquality("x1**2 + x2**2 - 1")
    df = pd.DataFrame(
        {
            "x1": [0.6, 0.5, 1],
            "x2": [0.8, 0.5, 1],
            "x3": [1, 1, 3],
        }
    )

    assert np.allclose(constraint.eval(df), [0, -0.5, 1])
    assert np.allclose(constraint.satisfied(df), [True, False, False])

    eval(constraint.__repr__())
    json.dumps(constraint.to_config())
    constraint2 = make_constraint(**constraint.to_config())
    assert np.allclose(constraint2.eval(df), constraint.eval(df))


def test_nonlinear_inequality():
    constraint = NonlinearInequality("x1**2 + x2**2 - 1")
    df = pd.DataFrame(
        {
            "x1": [0.6, 0.5, 1],
            "x2": [0.8, 0.5, 1],
            "x3": [1, 1, 3],
        }
    )

    assert np.allclose(constraint.eval(df), [0, -0.5, 1])
    assert np.allclose(constraint.satisfied(df), [True, True, False])

    eval(constraint.__repr__())
    json.dumps(constraint.to_config())
    constraint2 = make_constraint(**constraint.to_config())
    assert np.allclose(constraint2.eval(df), constraint.eval(df))


def test_nchoosek():
    names = ["x1", "x2", "x3", "x4"]
    constraint = NChooseK(names=names, max_active=3)

    df = pd.DataFrame([[1, 0, 1, 1], [1, 2, 4, 4]], columns=names)

    assert np.allclose(constraint.satisfied(df), [True, False])

    eval(constraint.__repr__())
    json.dumps(constraint.to_config())
    make_constraint(**constraint.to_config())


def test_constraints():
    constraints = Constraints(
        [
            LinearEquality(["x1", "x2", "x3"], lhs=[1, 1, 1], rhs=1),
            LinearInequality(["x1", "x2"], lhs=[1, 1], rhs=0.5),
            NonlinearEquality("x1 + x2 + x3 - 1"),
            NonlinearInequality("x1**2 + x3**2 - 1"),
            NChooseK(["x1", "x2"], max_active=2),
        ]
    )

    df = pd.DataFrame({"x1": [0.1, 0.3], "x2": [0.2, 0.3], "x3": [0.7, 0.4]})
    assert np.allclose(constraints.satisfied(df), [True, False])

    # serialize
    config = constraints.to_config()
    constraints = Constraints(config)
    assert np.allclose(constraints.satisfied(df), [True, False])

    # skip empty constraints
    config = [{"type": "linear-equality", "names": [], "lhs": []}]
    constraints = Constraints(config)
    assert len(constraints) == 0


def test_make_constraint():
    with pytest.raises(ValueError):
        make_constraint(type="box-bound")  # unknown constraint type
