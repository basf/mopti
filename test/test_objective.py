import json

import numpy as np
import pandas as pd

import opti
from opti.objective import CloseToTarget, Maximize, Minimize, Objectives, make_objective


def make_dataframe():
    return pd.DataFrame(
        {
            "meetings": [10, 21, 10, 1, 2],
            "coffee": [2, 3, 5, 7, 1],
            "seriousness": [5, 8, 2, 4, 10],
        },
        index=["monday", "tuesday", "wednesday", "thursday", "friday"],
    )


def test_minimize():
    obj = Minimize("meetings")
    eval(obj.__repr__())
    json.dumps(obj.to_config())
    make_objective(**obj.to_config())
    assert obj.name == "meetings"

    s = make_dataframe()["meetings"]
    assert np.allclose(obj(s), s)


def test_maximize():
    obj = Maximize("coffee")
    eval(obj.__repr__())
    json.dumps(obj.to_config())
    make_objective(**obj.to_config())
    assert obj.name == "coffee"

    s = make_dataframe()["meetings"]
    assert np.allclose(obj(s), s * (-1))


def test_closetotarget():
    obj = CloseToTarget("seriousness", target=5, exponent=2)
    eval(obj.__repr__())
    json.dumps(obj.to_config())
    obj2 = make_objective(**obj.to_config())
    assert obj2.name == "seriousness"
    assert obj2.target == 5
    assert obj2.exponent == 2

    s = make_dataframe()["meetings"]
    assert np.allclose(obj(s), (s - 5) ** 2)


def test_objectives():
    objectives = Objectives(
        [
            Minimize("meetings", target=2),
            Maximize("coffee", target=4),
            CloseToTarget("seriousness", target=5, exponent=2, tolerance=1.1),
        ]
    )

    objectives.__repr__()
    json.dumps(objectives.to_config())

    assert objectives.names == [
        "meetings",
        "coffee",
        "seriousness",
    ]

    df = make_dataframe()
    Z = objectives(df)
    assert len(Z) == len(df)
    assert np.all(Z.columns == objectives.names)

    objs = objectives.get(Minimize)
    assert isinstance(objs, Objectives)
    assert len(objs) == 1
    for o in objs:
        assert isinstance(o, Minimize)

    objs = objectives.get((Maximize, CloseToTarget))
    assert isinstance(objs, Objectives)
    assert len(objs) == 2
    for o in objs:
        assert isinstance(o, (Maximize, CloseToTarget))


def test_ideal_nadir():
    # problem with maximize objectives
    problem = opti.problems.Detergent()
    bounds = problem.objectives.bounds(problem.outputs)
    assert np.allclose(bounds.loc["min"], -3)
    assert np.allclose(bounds.loc["max"], 0)

    # problem with close-to-target objective
    problem = opti.problems.Cake()
    bounds = problem.objectives.bounds(problem.outputs)
    assert np.allclose(bounds.loc["min"], [300, -5, 0])
    assert np.allclose(bounds.loc["max"], [600, 0, 1.4])
