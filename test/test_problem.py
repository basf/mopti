import json

import numpy as np
import pandas as pd
import pytest

import opti
from opti.model import LinearModel
from opti.objective import Minimize
from opti.parameter import Continuous, Parameters
from opti.problem import Problem


def test_parameters():
    # set inputs & outputs as Parameters object
    problem = opti.Problem(
        inputs=Parameters([opti.Continuous("x")]),
        outputs=Parameters([opti.Continuous("y")]),
    )
    assert isinstance(problem.inputs, opti.Parameters)
    assert isinstance(problem.outputs, opti.Parameters)

    # set inputs & outputs as a list of Parameter
    problem = opti.Problem(
        inputs=[opti.Continuous("x", domain=[0, 1])],
        outputs=[opti.Continuous("y", domain=[0, 1])],
    )
    assert isinstance(problem.inputs, opti.Parameters)
    assert isinstance(problem.outputs, opti.Parameters)

    # set inputs & outputs as list of dicts
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


def test_properties():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(10)],
        outputs=[opti.Continuous(f"y{i}") for i in range(4)],
        objectives=[opti.Minimize(f"y{i}") for i in range(2)],
    )
    assert problem.n_inputs == 10
    assert problem.n_outputs == 4
    assert problem.n_objectives == 2
    assert problem.n_constraints == 0

    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}") for i in range(10)],
        outputs=[opti.Continuous(f"y{i}") for i in range(4)],
        constraints=[opti.LinearEquality([f"x{i}" for i in range(10)], rhs=1)],
    )
    assert problem.n_inputs == 10
    assert problem.n_outputs == 4
    assert problem.n_objectives == 4
    assert problem.n_constraints == 1


def test_json(tmpdir):
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

    # serialize to json
    tmpfile = tmpdir.join("foo.json")
    problem = opti.Problem.from_json("examples/bread.json")
    problem.to_json(tmpfile)

    # json compliance with respect to Inf and Nan
    problem = opti.problems.Ackley()
    problem.outputs["y"].low = 0  # output domain is now [0, Inf]
    problem.to_json(tmpfile)

    conf = json.load(open(tmpfile))
    assert conf["outputs"][0]["domain"][1] is None

    problem = opti.Problem.from_json(tmpfile)
    assert problem.outputs.bounds.loc["min", "y"] == 0
    assert problem.outputs.bounds.loc["max", "y"] == np.inf

    # handling of unicode characters
    problem = opti.Problem(
        inputs=[opti.Continuous("öäüßèé", [0, 1])],
        outputs=[opti.Continuous("/[].#@²³$")],
    )
    problem.to_json(tmpfile)
    problem2 = opti.read_json(tmpfile)
    assert np.all(problem2.inputs.names == problem.inputs.names)
    assert np.all(problem2.outputs.names == problem.outputs.names)

    # parse values scientific notation
    problem = opti.read_json("examples/test_scientific_notation.json")
    assert np.allclose(problem.data["x"], [0.9, 1.7e12])


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
    # test with continuous and discrete parameters
    problem = opti.Problem(
        inputs=[opti.Discrete("x", [0, 1, 2])], outputs=[opti.Continuous("y")]
    )

    # this works
    problem.set_data(data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}))

    # missing parameter in data
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"y": [0, 1, 2]}))
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1, 2]}))

    # missing value in inputs
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1, None], "y": [0, 1, 2]}))

    # missing output values in outputs are ok
    problem.set_data(data=pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, None]}))

    # outputs with all missing values are not ok
    with pytest.raises(ValueError):
        problem.set_data(data=pd.DataFrame({"x": [0, 1], "y": [None, None]}))

    # non-numeric value for continuous / discrete parameter
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": ["oops", 1, 2], "y": [0, 1, 2]}))
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": [0, 1, 2], "y": ["oops", 1, 2]}))

    # test with categorical parameter
    problem = opti.Problem(
        inputs=[opti.Categorical("x", ["A", "B"])], outputs=[opti.Continuous("y")]
    )

    # this works
    problem.set_data(data=pd.DataFrame({"x": ["A", "B", "A"], "y": [0, 1, 0.5]}))

    # categorical levels in data must be known
    with pytest.raises(ValueError):
        problem.set_data(pd.DataFrame({"x": ["A", "B", "C", None], "y": [0, 1, 2, 3]}))

    # test the case where a data frame with numeric values is passed for a categorical parameter
    problem = opti.Problem(
        inputs=[{"name": "x", "type": "categorical", "domain": ["0", "1", "2"]}],
        outputs=[{"name": "y", "type": "continuous", "domain": [0, 1]}],
    )
    problem.set_data(pd.DataFrame({"x": [0, 1, 2], "y": [0.1, 0.7, 0.3]}))


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
    # test handling of empty constraints
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

    config = {
        "inputs": [{"type": "continuous", "name": "x"}],
        "outputs": [{"type": "continuous", "name": "y"}],
        "constraints": [
            {"names": [], "lhs": [], "rhs": 1, "type": "linear-inequality"}
        ],
    }
    problem = opti.Problem.from_config(config)
    assert problem.constraints is None


def test_config():
    # test if the configuration dict is strictly json-compliant
    problem = opti.problems.Ackley()
    problem.create_initial_data(2)
    problem.data.loc[0, "y"] = np.nan
    problem.outputs["y"].low = 0  # output domain is now [0, Inf]

    conf = problem.to_config()
    # the upper bound np.inf is converted to None
    assert conf["outputs"][0]["domain"][1] is None
    # the datapoint y = np.nan is converted to None
    assert conf["data"]["data"][0][-1] is None


def test_optima():
    problem = opti.read_json("examples/simple.json")
    assert len(problem.optima) == 2


def test_models():
    D = 5
    M = 3
    problem = Problem(
        inputs=[Continuous(f"x{i}", [0, 1]) for i in range(D)],
        outputs=[Continuous(f"y{i}") for i in range(M)],
        objectives=[Minimize(f"y{i}") for i in range(M - 1)],
        models=[
            LinearModel([f"y{i}"], coefficients={f"x{i}": 1 for i in range(D)})
            for i in range(M)
        ],
    )

    X = problem.sample_inputs(10)
    Y = problem.models(X)
    Z = problem.objectives(Y)
    assert list(Y.columns) == problem.outputs.names
    assert list(Z.columns) == problem.objectives.names
