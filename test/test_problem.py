import json

import numpy as np
import pandas as pd
import pytest

import opti
from opti.model import LinearModel
from opti.objective import Minimize
from opti.parameter import Continuous
from opti.problem import Problem


def test_parameters():
    # test if inputs / outputs can be specified as list of Dimensions
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", domain=[0, 1]) for i in range(3)],
        outputs=[opti.Continuous(f"y{i}", domain=[0, 1]) for i in range(3)],
    )
    assert isinstance(problem.inputs, opti.Parameters)
    assert isinstance(problem.outputs, opti.Parameters)

    # test if inputs / outputs can be specified as list of Dimensions
    problem = opti.Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "categorical", "domain": ["A", "B", 3]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
    )
    assert isinstance(problem.inputs, opti.Parameters)
    assert isinstance(problem.outputs, opti.Parameters)


def test_problem():
    # test loading from json
    problem = opti.read_json("examples/bread.json")
    assert len(problem.inputs) == 11
    assert len(problem.outputs) == 4
    assert problem.constraints is not None
    assert problem.name is not None

    data = pd.DataFrame(
        columns=problem.inputs.names + problem.outputs.names,
        data=np.zeros((3, len(problem.inputs) + len(problem.outputs))),
    )
    data["yeast_type"] = "A"
    problem.set_data(data)

    # test repr/str
    problem.__str__()
    problem.__repr__()

    # objectives are optional and should be set to minimize if not given
    config = json.load(open("examples/bread.json"))
    config.pop("objectives")
    problem = opti.Problem.from_config(config)
    assert len(problem.objectives) == len(problem.outputs)
    for obj in problem.objectives:
        assert isinstance(obj, opti.objective.Minimize)

    # constraints are optional
    config = json.load(open("examples/bread.json"))
    config.pop("constraints")
    problem = opti.Problem.from_config(config)
    assert problem.constraints is None


def test_inconsistent_parameters():
    # duplicate parameter name in inputs and outputs
    with pytest.raises(ValueError):
        opti.Problem(inputs=[opti.Continuous("x")], outputs=[opti.Continuous("x")])


def test_inconsistent_objectives():
    # parameter name in objective not among inputs & outputs
    with pytest.raises(ValueError):
        opti.Problem(
            inputs=[opti.Continuous("x")],
            outputs=[opti.Continuous("y")],
            objectives=[opti.objective.Minimize("z")],
        )


def test_set_data():
    problem = opti.Problem(
        inputs=[opti.Discrete("x", [0, 1, 2])], outputs=[opti.Continuous("y")]
    )
    problem.set_data(data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}))

    # missing input
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"y": [0, 1, 2]}))

    # missing output
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1, 2]}))

    # missing value in inputs
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1, np.nan], "y": [0, 1, 2]}))

    # no value in outputs
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1], "y": [np.nan, np.nan]}))

    # non-numeric value for continuous parameter
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": ["oops", 1, 2], "y": [0, 1, 2]}))
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": [0, 1, 2], "y": ["oops", 1, 2]}))

    # test with categorical parameter
    problem = opti.Problem(
        inputs=[opti.Categorical("x", ["A", "B"])], outputs=[opti.Continuous("y")]
    )

    problem.set_data(data=pd.DataFrame({"x": ["A", "A"], "y": [0, 1]}))

    # unknown category
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": ["A", "B", "C"], "y": [0, 1, 2]}))


def test_add_data():
    problem = opti.Problem(
        inputs=[opti.Continuous("x")], outputs=[opti.Continuous("y")]
    )

    # adding data when no data exists
    problem.add_data(data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    assert len(problem.data) == 2

    # adding new data with extra column
    problem.add_data(data=pd.DataFrame({"x": [2], "y": [2], "z": [2]}))
    assert len(problem.data) == 3
    assert "z" in problem.data.columns

    # adding data with missing input value
    with pytest.raises(ValueError):
        problem.add_data(data=pd.DataFrame({"x": [np.nan], "y": [3]}))

    # adding data with missing input value
    with pytest.raises(ValueError):
        problem.add_data(data=pd.DataFrame({"x": [3]}))


def test_get_XY():
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Discrete("x2", [0, 1, 5]),
            opti.Categorical("x3", ["A", "B", "C"]),
        ],
        outputs=[opti.Continuous("y1"), opti.Continuous("y2")],
        data=pd.DataFrame(
            {
                "x1": [0.2, 0.5, 0.6],
                "x2": [0, 1, 5],
                "x3": ["A", "C", "B"],
                "y1": [1, 0.8, 0.4],
                "y2": [0, 0.2, np.nan],
            }
        ),
    )

    X = problem.get_X()
    assert X.shape == (3, 3)
    Y = problem.get_Y()
    assert Y.shape == (3, 2)

    # nans are dropped
    X, Y = problem.get_XY()
    assert X.shape == (2, 3)
    assert Y.shape == (2, 2)

    # using transforms
    X, Y = problem.get_XY(
        continuous="normalize", discrete="normalize", categorical="onehot-encode"
    )
    assert X.shape == (2, 5)
    assert Y.shape == (2, 2)

    # consider subset of outputs
    X, Y = problem.get_XY(outputs=["y1"])
    assert X.shape == (3, 3)
    assert Y.shape == (3, 1)

    # pass data
    data = pd.DataFrame(
        {
            "x1": [0, 0, 0],
            "x2": [0, 0, 0],
            "x3": ["A", "B", "C"],
            "y1": [1, np.nan, 0.4],
            "y2": [0, 0.2, np.nan],
        }
    )
    X, Y = problem.get_XY(data=data)
    assert X.shape == (1, 3)
    assert Y.shape == (1, 2)


def test_X_bounds():
    # Obtaining input data bounds should work for continuous and discrete parameters.
    # For parameters with 0 variation, require xhi = xlo + 1.
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [-1, 1]),
            opti.Continuous("x2", [-1, 1]),
            opti.Discrete("x3", [0, 1, 2]),
        ],
        outputs=[opti.Continuous("y")],
        data=pd.DataFrame({"x1": [-0.2, 0.5], "x2": [0, 0], "x3": [2, 0], "y": [0, 1]}),
    )
    xlo, xhi = problem.get_X_bounds()
    assert np.allclose(xlo, [-0.2, 0, 0])
    assert np.allclose(xhi, [0.5, 1, 2])


def test_empty_constraints():
    problem = opti.Problem(
        inputs=[opti.Continuous("x")],
        outputs=[opti.Continuous("y")],
        constraints=[],
    )
    assert problem.constraints is None

    config = {
        "inputs": [{"type": "continuous", "name": "x", "domain": [0, 1]}],
        "outputs": [{"type": "continuous", "name": "y", "domain": [0, 1]}],
        "constraints": [],
    }
    problem = opti.Problem.from_config(config)
    assert problem.constraints is None


def test_eval():
    problem = opti.problems.ZDT1(3)
    X = problem.inputs.sample(2)
    Y = problem.eval(X)
    assert len(Y) == 2
    assert (X.index == Y.index).all()


def test_config():
    # test if the configuration dict is strictly json-compliant
    problem = opti.problems.Ackley()
    problem.create_initial_data(2)
    problem.data.loc[0, "y"] = np.nan

    conf = problem.to_config()
    # domain [-np.inf, np.inf] should be converted to [None, None]
    assert conf["outputs"][0]["domain"][0] is None
    assert conf["outputs"][0]["domain"][1] is None
    # the the datapoint y = np.nan should be converted to None
    assert conf["data"]["data"][0][-1] is None


def test_json(tmpdir):
    tmpfile = tmpdir.join("foo.json")

    # test if problems are serializable to/from json
    problem = opti.Problem.from_json("examples/bread.json")
    problem.to_json(tmpfile)

    # test json compliance
    problem = opti.problems.Ackley()  # output domain is [-Inf, Inf]
    problem.to_json(tmpfile)

    conf = json.load(open(tmpfile))
    assert conf["outputs"][0]["domain"][0] is None
    assert conf["outputs"][0]["domain"][1] is None

    problem = opti.Problem.from_json(tmpfile)
    assert problem.outputs.bounds.loc["min", "y"] == -np.inf
    assert problem.outputs.bounds.loc["max", "y"] == np.inf

    # test unicode characters
    problem = opti.Problem(
        inputs=[opti.Continuous("öäüßèé", [0, 1])],
        outputs=[opti.Continuous("/[].#@²³$")],
    )
    problem.to_json(tmpfile)
    problem2 = opti.read_json(tmpfile)
    assert np.all(problem2.inputs.names == problem.inputs.names)
    assert np.all(problem2.outputs.names == problem.outputs.names)


def test_qritos():
    # test if empty constraints coming from Qritos are ignored
    problem = opti.Problem.from_config(
        {
            "inputs": [{"type": "continuous", "name": "x"}],
            "outputs": [{"type": "continuous", "name": "y"}],
            "constraints": [
                {"names": [], "lhs": [], "rhs": 1, "type": "linear-inequality"}
            ],
        }
    )
    assert problem.constraints is None


def test_optima():
    problem = opti.read_json("examples/simple.json")
    front = problem.optima
    assert len(front) == 2
    problem.set_optima(front)


def test_models():
    D = 5
    M = 3
    problem = Problem(
        inputs=[Continuous(f"x{i}", [0, 1]) for i in range(D)],
        outputs=[Continuous(f"y{i}") for i in range(M)],
        objectives=[Minimize(f"y{i}") for i in range(M - 1)],
        models=[LinearModel([f"y{i}"], coefficients=np.ones(D)) for i in range(M)],
    )

    X = problem.sample_inputs(10)
    Y = problem.models.eval(X)
    Z = problem.objectives.eval(Y)
    assert list(Y.columns) == problem.outputs.names
    assert list(Z.columns) == problem.objectives.names
