import numpy as np
import pytest

from opti import Problem
from opti.constraint import (
    Constraint,
    Constraints,
    LinearEquality,
    LinearInequality,
    NonlinearEquality,
    NonlinearInequality,
)
from opti.parameter import Categorical, Continuous, Discrete, Parameter, Parameters
from opti.tools.reduce import (
    ReducedProblem,
    check_existence_of_solution,
    find_continuous_inputs,
    find_linear_equality_constraints,
    reduce,
    remove_eliminated_inputs,
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


# def test_reduce()
# teste fall: nur nullconstraint
# EINBAUEN: WAS PASSIERT MIT DOMAIN [NONE, X], [X, NONE]?
# TESTE: KANN EIN REDUZIERTES PROBLEM WIEDER REDUZIERT WERDEN?
# --> übernehme zusätzlich alte equalities, oder: werfe NotImplementedError
# PROBLEM: Alles, was nicht linear inequality ist, bleibt bestehen --> problem, da zum teil Variablen verschwinden
# --> nachdem neues problem aufgebaut wurde: def refresh_constraints


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
    _problem = reduce(problem)

    assert len(_problem.inputs) == 3
    assert len(_problem.constraints) == 2
    lhs = [-1.0, 1.0]
    rhs = [2.0, -1.0]
    for i, c in enumerate(_problem.constraints):
        assert isinstance(c, LinearInequality)
        assert c.names == ["x2"]
        assert c.lhs == [lhs[i]]
        assert c.rhs == rhs[i]

    # assert _problem._equalities == [["x1", ["x2"], [-1.0, 0.0]]]

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
    assert problem == reduce(problem)

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
        reduce(problem)

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
        reduce(problem)

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
        reduce(problem)


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
    _problem = reduce(problem)

    assert len(_problem.inputs) == 1
    lhs = [-1, 1]
    rhs = [1, 1]
    for i, c in enumerate(_problem.constraints):
        assert c.names == ["x3"]
        assert c.lhs == [lhs[i]]
        assert c.rhs == rhs[i]
    assert _problem._equalities == [["x1", ["x3"], [-1.0, 0.0]], ["x2", [], [1.0]]]


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
        reduce(problem)


def test_remove_eliminated_inputs():
    # define problem: with nonlinear Equality
    problem = ReducedProblem(
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
        _equalities=[["x1", ["x2"], [-1.0, 0.0]]],
    )
    with pytest.warns(UserWarning):
        remove_eliminated_inputs(problem)

    # define problem: no solution for constraints
    problem = ReducedProblem(
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
        _equalities=[["x1", ["x2"], [-1.0, 0.0]]],
    )
    with pytest.raises(RuntimeError):
        remove_eliminated_inputs(problem)

    # define problem: linear equality can be removed (is always fulfilled)
    # From: reduce(problem)
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

    problem = ReducedProblem(
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
        _equalities=[["x1", ["x2"], [-1.0, 0.0]]],
    )
    _problem = remove_eliminated_inputs(problem)
    assert len(_problem.constraints) == 2

    # define problem: linear equality cant
    problem = ReducedProblem(
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
        _equalities=[["x1", ["x2"], [-1.0, 0.0]]],
    )
    _problem = remove_eliminated_inputs(problem)
    assert len(_problem.constraints) == 3
    assert _problem.constraints[0].names == ["x2"]
    assert _problem.constraints[0].lhs == [1.0]
    assert _problem.constraints[0].rhs == 1.0

    # define problem: larger problem
    # From reduce(problem)
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
    problem = ReducedProblem(
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
        _equalities=[["x1", ["x2", "x3"], [-1.0, -1.0, 1.0]]],
    )
    _problem = remove_eliminated_inputs(problem)

    assert len(_problem.constraints) == 3
    assert _problem.constraints[2].names == ["x2"]
    assert _problem.constraints[2].lhs == [-1.0]
    assert _problem.constraints[2].rhs == -2.0


def test_ReducedProblem_from_json():
    # define reduced problem
    problem = ReducedProblem(
        name="A simple problem.",
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
        _equalities=[["x1", ["x2", "x3"], [-1.0, -1.0, 1.0]]],
    )
    _problem = ReducedProblem.from_json("examples/simpleReducedProblem.json")
    assert _problem._equalities == problem._equalities


def test_ReducedProblem_invalid_equalities():
    # define problem: everything ok
    ReducedProblem(
        inputs=Parameters(
            [
                Continuous("x2", [-1, 1]),
                Continuous("x3", [-1, 1]),
            ]
        ),
        outputs=Parameters([Continuous("y1")]),
        _equalities=[["x1", ["x2", "x3"], [-1.0, -1.0, 1.0]]],
    )

    # define problem: invalid name in equalities
    with pytest.raises(ValueError):
        ReducedProblem(
            inputs=Parameters(
                [
                    Continuous("x2", [-1, 1]),
                    Continuous("x3", [-1, 1]),
                ]
            ),
            outputs=Parameters([Continuous("y1")]),
            _equalities=[["x1", ["a", "x3"], [-1.0, -1.0, 1.0]]],
        )

    # define problem: invalid coefficient in equalities (inf)
    with pytest.raises(ValueError):
        ReducedProblem(
            inputs=Parameters(
                [
                    Continuous("x2", [-1, 1]),
                    Continuous("x3", [-1, 1]),
                ]
            ),
            outputs=Parameters([Continuous("y1")]),
            _equalities=[["x1", ["x2", "x3"], [-1.0, -np.inf, 1.0]]],
        )

    # define problem: invalid coefficient in equalities (inf)
    with pytest.raises(ValueError):
        ReducedProblem(
            inputs=Parameters(
                [
                    Continuous("x2", [-1, 1]),
                    Continuous("x3", [-1, 1]),
                ]
            ),
            outputs=Parameters([Continuous("y1")]),
            _equalities=[["x1", ["x2", "x3"], [-1.0, np.nan, 1.0]]],
        )


def test_ReducedProblem_augment_data():
    problem = Problem.from_json("examples/bread.json")
    data = problem.data

    _problem = reduce(problem)
    names = np.concatenate((_problem.inputs.names, problem.outputs.names))
    _problem.data = data[names]

    data_rec = ReducedProblem.augment_data(_problem.data, _problem._equalities)

    assert [column in data.columns for column in data_rec.columns]
    assert [column in data_rec.columns for column in data.columns]
    for col in data.columns:
        data[col].eq(data_rec[col])
