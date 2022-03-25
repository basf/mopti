import numpy as np
import pandas as pd
import pytest

from opti import Problem, read_json
from opti.constraint import (
    Constraint,
    Constraints,
    LinearEquality,
    LinearInequality,
    NonlinearEquality,
    NonlinearInequality,
)
from opti.parameter import Categorical, Continuous, Discrete, Parameter, Parameters
from opti.tools.reduce_problem import (
    AffineTransform,
    check_existence_of_solution,
    find_continuous_inputs,
    find_linear_equality_constraints,
    reduce_problem,
    remove_eliminated_inputs,
    rref,
)


def test_find_linear_constraints():
    # define Constraints
    constraints = Constraints(
        [
            Constraint(),
            LinearEquality(["x", "y"], [1, 2], 3),
            LinearInequality(["a", "b"], [4, 5], 6),
            NonlinearEquality("x**2 + y**2 - 1"),
            NonlinearInequality("a**4 + b**4 - 1"),
        ]
    )
    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        constraints
    )

    assert isinstance(linearEqualityConstraints, Constraints)
    assert len(linearEqualityConstraints) == 1
    assert isinstance(linearEqualityConstraints.constraints[0], LinearEquality)

    assert isinstance(otherConstraints, Constraints)
    assert len(otherConstraints) == 4
    for c in otherConstraints:
        assert not isinstance(c, LinearEquality)

    # define Constraints
    constraints = Constraints([])
    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        constraints
    )

    assert isinstance(linearEqualityConstraints, Constraints)
    assert len(linearEqualityConstraints) == 0

    assert isinstance(otherConstraints, Constraints)
    assert len(otherConstraints) == 0

    # define constraints
    constraints = Constraints(
        [LinearEquality(["x", "y"], [1, 2], 3), LinearEquality(["a", "b"], [4, 5], 6)]
    )

    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        constraints
    )

    assert isinstance(linearEqualityConstraints, Constraints)
    assert len(linearEqualityConstraints) == 2
    for c in linearEqualityConstraints:
        assert isinstance(c, LinearEquality)

    assert isinstance(otherConstraints, Constraints)
    assert len(otherConstraints) == 0

    # define constraints
    constraints = Constraints(
        [
            Constraint(),
            NonlinearEquality("x**2 + y**2 - 1"),
            NonlinearInequality("a**4 + b**4 - 1"),
        ]
    )
    linearEqualityConstraints, otherConstraints = find_linear_equality_constraints(
        constraints
    )

    assert isinstance(linearEqualityConstraints, Constraints)
    assert len(linearEqualityConstraints) == 0

    assert isinstance(otherConstraints, Constraints)
    assert len(otherConstraints) == 3
    for c in otherConstraints:
        assert not isinstance(c, LinearEquality)


def test_find_continuous_inputs():
    # define input parameters
    inputs = Parameters(
        [
            Parameter("n0", [None, None]),
            Continuous("n1", [0, 1]),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3", ["A", "B"]),
            Continuous("n4", [1, 2]),
        ]
    )
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 2
    for p in contInputs:
        assert isinstance(p, Continuous)

    assert isinstance(otherInputs, Parameters)
    assert len(otherInputs) == 3
    for p in otherInputs:
        assert not isinstance(p, Continuous)

    # define input parameters
    inputs = Parameters(
        [
            Parameter("n0", [None, None]),
            Continuous("n1", [0, 1]),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3", ["A", "B"]),
            Continuous("n4", [1, 2]),
        ]
    )
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 2
    for p in contInputs:
        assert isinstance(p, Continuous)

    assert isinstance(otherInputs, Parameters)
    assert len(otherInputs) == 3
    for p in otherInputs:
        assert not isinstance(p, Continuous)

    # define input parameters
    inputs = Parameters([])
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 0

    assert isinstance(otherInputs, Parameters)
    assert len(otherInputs) == 0

    # define input parameters
    inputs = Parameters(
        [
            Continuous("n1", [0, 1]),
            Continuous("n2", [1, 2]),
            Continuous("n3", [2, 4]),
        ]
    )
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 3
    for p in contInputs:
        assert isinstance(p, Continuous)

    assert isinstance(otherInputs, Parameters)
    assert len(otherInputs) == 0

    # define input parameters
    inputs = Parameters(
        [
            Parameter("n1", [None, None]),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3", ["A", "B"]),
        ]
    )
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 0

    assert isinstance(otherInputs, Parameters)
    assert len(otherInputs) == 3
    for p in otherInputs:
        assert not isinstance(p, Continuous)


def check_problem_for_reduction():
    # define test problem: no constraints
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "categorical", "domain": ["A", "B"]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
    )
    assert not check_problem_for_reduction(problem)

    # define test problem: no linear equality constraints
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "categorical", "domain": ["A", "B"]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints(
            [
                Constraint(),
                NonlinearEquality("x1**2 + x2**2 - 1"),
                NonlinearInequality("x1**4 + x2**4 - 1"),
                LinearInequality(["x1", "x2"], [4, 5], 6),
            ]
        ),
    )
    assert not check_problem_for_reduction(problem)

    # define test problem: no continuous input
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "discrete", "domain": [1, 10]},
            {"name": "x2", "type": "categorical", "domain": ["A", "B"]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints([LinearInequality(["x1", "x2"], [4, 5], 6)]),
    )
    assert not check_problem_for_reduction(problem)

    # define test problem: invalid linear inequalities
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "categorical", "domain": ["A", "B"]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints([LinearInequality(["x1", "x2"], [4, 5], 6)]),
    )
    with pytest.raises(RuntimeError):
        check_problem_for_reduction(problem)

    # define test problem: everything okay
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "continuous", "domain": [1, 3]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints([LinearInequality(["x1", "x2"], [4, 5], 6)]),
    )
    assert check_problem_for_reduction(problem)


def test_check_existence_of_solution():
    # rk(A) = rk(A_aug) = 1, 3 inputs
    A_aug = np.array([[1, 0, 0, -0.5], [1, 0, 0, -0.5]])
    check_existence_of_solution(A_aug)

    # rk(A) = 1, rk(A_aug) = 2, 3 inputs
    A_aug = np.array([[1, 0, 0, 0.5], [1, 0, 0, -0.5]])
    with pytest.raises(Warning):
        check_existence_of_solution(A_aug)

    # rk(A) = rk(A_aug) = 2, 3 inputs
    A_aug = np.array([[1, 0, 0, -0.5], [0, 1, 0, -0.5]])
    check_existence_of_solution(A_aug)

    # rk(A) = rk(A_aug) = 0, 3 inputs
    A_aug = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    check_existence_of_solution(A_aug)

    # rk(A) = rk(A_aug) = 2, 2 inputs
    A_aug = np.array([[1, 0, -0.5], [0, 1, -0.5]])
    with pytest.raises(Warning):
        check_existence_of_solution(A_aug)


def test_reduce_1_independent_linear_equality_constraints():
    # define problem: standard case
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [1, 2]),
                Continuous("x2", [-1, 1]),
                Categorical("x3", ["A", "B"]),
                Discrete("x4", [-1, 0, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
                LinearEquality(names=["x1", "x2"], lhs=[-0.5, -0.5], rhs=0),
                LinearInequality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
            ]
        ),
    )
    _problem, transform = reduce_problem(problem)

    assert len(_problem.inputs) == 3
    assert len(_problem.constraints) == 2
    lhs = [-1.0, 1.0]
    rhs = [2.0, -1.0]
    for i, c in enumerate(_problem.constraints):
        assert isinstance(c, LinearInequality)
        assert c.names == ["x2"]
        assert c.lhs == [lhs[i]]
        assert c.rhs == rhs[i]

    # define problem: irreducible problem
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [1, 2]),
                Continuous("x2", [-1, 1]),
                Categorical("x3", ["A", "B"]),
                Discrete("x4", [-1, 0, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints([]),
    )
    assert problem == reduce_problem(problem)[0]

    # define problem: invalid constraint (nonexisting name in linear equality constraint)
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [-1, 2]),
                Continuous("x2", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["a", "x2"], lhs=[1, 1], rhs=0),
                LinearEquality(names=["x1", "x2"], lhs=[-0.5, -0.5], rhs=0),
            ]
        ),
    )
    with pytest.raises(RuntimeError):
        reduce_problem(problem)

    # define problem: invalid constraint (non-continuous parameter in linear equality constraint)
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [-1, 2]),
                Continuous("x2", [-1, 1]),
                Categorical("x3", ["A", "B"]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["x3", "x2"], lhs=[1, 1], rhs=0),
                LinearEquality(names=["x1", "x2"], lhs=[-0.5, -0.5], rhs=0),
            ]
        ),
    )
    with pytest.raises(RuntimeError):
        reduce_problem(problem)

    # define problem: linear equality constraints can't be fulfilled inside the domain
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [1, 2]),
                Continuous("x2", [None, None]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["x1", "x2"], lhs=[1, 0], rhs=0),
            ]
        ),
    )
    with pytest.raises(Warning):
        reduce_problem(problem)


def test_reduce_2_independent_linear_equality_constraints():
    # define problem: standard case
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [-1, 1]),
                Continuous("x2", [-1, 1]),
                Continuous("x3", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["x1", "x2", "x3"], lhs=[1, 1, 1], rhs=1),
                LinearEquality(names=["x1", "x2", "x3"], lhs=[1, 2, 1], rhs=2),
                LinearEquality(names=["x1", "x2", "x3"], lhs=[-1, -1, -1], rhs=-1),
            ]
        ),
    )
    _problem, transform = reduce_problem(problem)

    assert len(_problem.inputs) == 1
    lhs = [-1, 1]
    rhs = [1, 1]
    for i, c in enumerate(_problem.constraints):
        assert c.names == ["x3"]
        assert c.lhs == [lhs[i]]
        assert c.rhs == rhs[i]
    assert transform.equalities == [["x1", ["x3"], [-1.0, 0.0]], ["x2", [], [1.0]]]


def test_reduce_3_independent_linear_equality_constraints():
    # define problem: standard case
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [-1, 1]),
                Continuous("x2", [-1, 1]),
                Continuous("x3", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(names=["x1", "x2", "x3"], lhs=[1, 1, 1], rhs=1),
                LinearEquality(names=["x1", "x2", "x3"], lhs=[1, 2, 1], rhs=2),
                LinearEquality(names=["x1", "x2", "x3"], lhs=[0, 0, 1], rhs=3),
            ]
        ),
    )
    with pytest.raises(Warning):
        reduce_problem(problem)


def test_remove_eliminated_inputs():
    # define problem: with nonlinear Equality
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearInequality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
                NonlinearEquality("x1**2 + x2**2 - 1"),
                LinearInequality(names=["x2"], lhs=[-1.0], rhs=2.0),
                LinearInequality(names=["x2"], lhs=[1.0], rhs=-1.0),
            ]
        ),
    )
    transform = AffineTransform(equalities=[["x1", ["x2"], [-1.0, 0.0]]])
    with pytest.warns(UserWarning):
        remove_eliminated_inputs(problem, transform)

    # define problem: no solution for constraints
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearInequality(names=["x1", "x2"], lhs=[1, 1], rhs=-1),
                LinearInequality(names=["x2"], lhs=[-1.0], rhs=2.0),
                LinearInequality(names=["x2"], lhs=[1.0], rhs=-1.0),
            ]
        ),
    )
    transform = AffineTransform(equalities=[["x1", ["x2"], [-1.0, 0.0]]])
    with pytest.raises(RuntimeError):
        remove_eliminated_inputs(problem, transform)

    # define problem: linear equality can be removed (is always fulfilled)
    # From: reduce_problem(problem)
    # problem = Problem(
    # inputs=Parameters(
    #    [
    #        Continuous("x1", [1, 2]),
    #        Continuous("x2", [-1, 1]),
    #    ]
    # ),
    # outputs=Parameters([Continuous("y1")]),
    # constraints=Constraints(
    #    [
    #        LinearEquality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
    #        LinearInequality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
    #        NonlinearEquality("x1**2 + x2**2 - 1"),
    #    ]
    # ),
    # )

    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearInequality(names=["x1", "x2"], lhs=[1, 1], rhs=0),
                LinearInequality(names=["x2"], lhs=[-1.0], rhs=2.0),
                LinearInequality(names=["x2"], lhs=[1.0], rhs=-1.0),
            ]
        ),
    )
    transform = AffineTransform(equalities=[["x1", ["x2"], [-1.0, 0.0]]])
    _problem = remove_eliminated_inputs(problem, transform)
    assert len(_problem.constraints) == 2

    # define problem: linear equality cant
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearInequality(names=["x1", "x2"], lhs=[1, 2], rhs=1),
                LinearInequality(names=["x2"], lhs=[-1.0], rhs=2.0),
                LinearInequality(names=["x2"], lhs=[1.0], rhs=-1.0),
            ]
        ),
    )
    transform = AffineTransform(equalities=[["x1", ["x2"], [-1.0, 0.0]]])
    _problem = remove_eliminated_inputs(problem, transform)
    assert len(_problem.constraints) == 3
    assert _problem.constraints[0].names == ["x2"]
    assert _problem.constraints[0].lhs == [1.0]
    assert _problem.constraints[0].rhs == 1.0

    # define problem: larger problem
    # From reduce_problem(problem)
    # problem = Problem(
    #    inputs=Parameters(
    #        [
    #            Continuous("x1", [-1, 1]),
    #            Continuous("x2", [-1, 1]),
    #            Continuous("x3", [-1, 1]),
    #        ]
    #    ),
    #    outputs=Parameters([Continuous("y1")]),
    #    constraints=Constraints(
    #        [
    #            LinearEquality(names=["x1", "x2", "x3"], lhs=[1, 1, 1], rhs=1),
    #        ]
    #    ),
    # )
    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
                Continuous("x3", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearInequality(names=["x2", "x3"], lhs=[-1.0, -1.0], rhs=0.0),
                LinearInequality(names=["x2", "x3"], lhs=[1.0, 1.0], rhs=2.0),
                LinearInequality(names=["x1", "x2", "x3"], lhs=[2, 1, 2], rhs=0.0),
            ]
        ),
    )
    transform = AffineTransform(equalities=[["x1", ["x2", "x3"], [-1.0, -1.0, 1.0]]])
    _problem = remove_eliminated_inputs(problem, transform)
    assert len(_problem.constraints) == 3
    assert _problem.constraints[2].names == ["x2"]
    assert _problem.constraints[2].lhs == [-1.0]
    assert _problem.constraints[2].rhs == -2.0


def test_AffineTransform_augment_data():
    problem = read_json("examples/bread.json")
    data = problem.data

    _problem, transform = reduce_problem(problem)
    names = np.concatenate((_problem.inputs.names, _problem.outputs.names))
    _problem.data = data[names]

    data_rec = transform.augment_data(_problem.data)

    assert all([column in data.columns for column in data_rec.columns])
    assert all([column in data_rec.columns for column in data.columns])
    data[["elasticity", "taste"]] = data[["elasticity", "taste"]].fillna(0)
    data_rec[["elasticity", "taste"]] = data_rec[["elasticity", "taste"]].fillna(0)
    for col in data.columns:
        assert all(data[col] == data_rec[col])


def test_AffineTransform_drop_data():
    # define problem and transform
    problem = read_json("examples/bread.json")
    data = problem.data

    _problem, transform = reduce_problem(problem)
    names = np.concatenate((_problem.inputs.names, _problem.outputs.names))

    data_drop = transform.drop_data(problem.data)
    _data = problem.data[names].copy()

    assert all([column in _data.columns for column in data_drop.columns])
    assert all([column in data_drop.columns for column in _data.columns])
    _data.loc[:, ["elasticity", "taste"]] = _data.loc[
        :, ["elasticity", "taste"]
    ].fillna(0)
    data_drop.loc[:, ["elasticity", "taste"]] = data_drop.loc[
        :, ["elasticity", "taste"]
    ].fillna(0)
    for col in _data.columns:
        assert all(_data[col] == data_drop[col])

    # the same situation again, but this time two columns have already been dropped
    data_drop = transform.drop_data(data.drop(columns=["elasticity", "taste"]))
    _data = data[names].drop(columns=["elasticity", "taste"])

    assert all([column in _data.columns for column in data_drop.columns])
    assert all([column in data_drop.columns for column in _data.columns])
    for col in _data.columns:
        assert all(_data[col] == data_drop[col])


def test_reduce_large_problem():
    def f(data: pd.DataFrame) -> pd.DataFrame:
        data["y1"] = data["x1"] + 2 * data["x2"] + data["x3"] + 2 * data["x4"]
        return data.loc[:, ["y1"]]

    data = pd.DataFrame(
        [
            [-0.5, 0.5, 1, 0, 1.5],
            [-1.0, 0.5, 1.0, 0.5, 2.0],
            [-1.0, 1.0, 0.0, 1.0, 3.0],
        ],
        columns=["x1", "x2", "x3", "x4", "y1"],
    )

    problem = Problem(
        inputs=Parameters(
            [
                Continuous("x1", [-1, 1]),
                Continuous("x2", [None, 1]),
                Continuous("x3", [None, None]),
                Continuous("x4", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        constraints=Constraints(
            [
                LinearEquality(["x1", "x2", "x4"], [1.0, -1.0, 1.0], -1.0),
                LinearEquality(["x2", "x3"], [2, 1], 2.0),
                LinearEquality(["x1", "x2", "x3", "x4"], [1.0, 1.0, 1.0, 1.0], 1.0),
                LinearInequality(["x1", "x2"], [1.0, 1.0], 1.0),
                LinearInequality(["x1", "x2", "x4"], [1.0, -1.0, 1.0], 0.0),
            ]
        ),
        f=f,
        data=data,
    )

    _problem, transform = reduce_problem(problem)

    assert all(data["y1"] == _problem.f(data)["y1"])
    assert transform.equalities == [
        ["x1", ["x3", "x4"], [-0.5, -1.0, 0.0]],
        ["x2", ["x3"], [-0.5, 1.0]],
    ]
    assert len(_problem.constraints) == 4

    assert all(_problem.constraints[0].names == np.array(["x3", "x4"]))
    assert all(_problem.constraints[0].lhs == [-1.0, -1.0])
    assert _problem.constraints[0].rhs == 0.0

    assert all(_problem.constraints[1].names == np.array(["x3", "x4"]))
    assert all(_problem.constraints[1].lhs == [-0.5, -1.0])
    assert _problem.constraints[1].rhs == 1.0

    assert all(_problem.constraints[2].names == np.array(["x3", "x4"]))
    assert all(_problem.constraints[2].lhs == np.array([0.5, 1.0]))
    assert _problem.constraints[2].rhs == 1.0

    assert _problem.constraints[3].names == ["x3"]
    assert _problem.constraints[3].lhs == [-0.5]
    assert _problem.constraints[3].rhs == 0.0

    _data = _problem.data[["x3", "x4", "y1"]]
    data_rec = transform.augment_data(_data)
    for col in data.columns:
        assert all(data[col] == data_rec[col])


def test_rref():
    # define A
    A = np.vstack(([[np.pi, 1e10, np.log(10), 7]], [[np.pi, 1e10, np.log(10), 7]]))
    A_rref, pivots = rref(A)
    B_rref = np.array(
        [
            [1.0, 3183098861.837907, 0.7329355988794278, 2.228169203286535],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([0]))

    # define A
    A = np.array(
        [
            [np.pi, 2, 1, 1, 1],
            [1e10, np.exp(0), 2, 2, 2],
            [np.log(10), -5.2, 3, 3, 3],
            [7, -3.5 * 1e-4, 4, 4, 4],
        ]
    )
    A_rref, pivots = rref(A)
    B_rref = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([0, 1, 2]))

    # define A
    A = np.ones(shape=(100, 100))
    A_rref, pivots = rref(A)
    B_rref = np.zeros(shape=(100, 100))
    B_rref[0, :] = 1

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([0]))

    # define A
    A = np.zeros(shape=(10, 20))
    A_rref, pivots = rref(A)
    B_rref = np.zeros(shape=(10, 20))

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([]))

    # define A
    A = np.array([[0, 1, 2, 2, 3, 4], [0, 5, 10, 6, 7, 8], [0, 9, 18, 10, 11, 12]])
    A_rref, pivots = rref(A)
    B_rref = np.array([[0, 1, 2, 0, -1, -2], [0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0]])

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([1, 3]))

    # define A
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    A_rref, pivots = rref(A)
    B_rref = np.array([[1, 0, -1, -2], [0, 1, 2, 3], [0, 0, 0, 0]])

    assert np.all(np.round(A_rref, 8) == np.round(B_rref, 8))
    assert all(np.array(pivots) == np.array([0, 1]))