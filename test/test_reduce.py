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
from opti.tools.reduce import find_continuous_inputs, find_linear_equality_constraints


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
            Parameter(),
            Continuous("n1", [0, 1]),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3"),
            Continuous("n4", [1, 2]),
        ]
    )
    contInputs, otherInputs = find_continuous_inputs(inputs)

    assert isinstance(contInputs, Parameters)
    assert len(contInputs) == 2
    for p in contInputs:
        assert isinstance(p, Continuous)

    assert isinstance(otherInputs, Parameters)
    assert len(contInputs) == 3
    for p in otherInputs:
        assert not isinstance(p, Continuous)

    # define input parameters
    inputs = Parameters(
        [
            Parameter(),
            Continuous("n1", [0, 1]),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3"),
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
            Parameter(),
            Discrete("n2", [0, 1, 2]),
            Categorical("n3"),
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
            {"name": "x2", "type": "categorical", "domain": ["A", "B", 3]},
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
            {"name": "x2", "type": "categorical", "domain": ["A", "B", 3]},
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
            {"name": "x2", "type": "categorical", "domain": ["A", "B", 3]},
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
            {"name": "x2", "type": "categorical", "domain": ["A", "B", 3]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints([LinearInequality(["x1", "x2"], [4, 5], 6)]),
    )
    assert not check_problem_for_reduction(problem)

    # define test problem: everything okay
    problem = Problem(
        inputs=[
            {"name": "x1", "type": "continuous", "domain": [1, 10]},
            {"name": "x2", "type": "continuous", "domain": ["A", "B", 3]},
        ],
        outputs=[
            {"name": "y1", "type": "continuous", "domain": [1, 10]},
            {"name": "y2", "type": "discrete", "domain": [1, 2, 3]},
        ],
        constraints=Constraints([LinearInequality(["x1", "x2"], [4, 5], 6)]),
    )
    assert check_problem_for_reduction(problem)


# def test_reduce()
# teste fall: nur nullconstraint
"""
import opti

problem = opti.Problem(
    inputs=[
        opti.Continuous("in_a", [0, 0.6]),
        opti.Continuous("in_b", [0, 0.75]),
        opti.Continuous("in_c", [0, 0.85]),
        opti.Continuous("in_d", [15, 40]),
        opti.Categorical("in_e", domain=["A", "B"]),
        opti.Discrete("in_f", [0, 4, 8]),
        opti.Continuous("in_g", [2, 5]),
        opti.Continuous("in_h", [0, np.log(3 + 1)]),
        opti.Continuous("in_i", [0, np.log(300 + 1)]),
        opti.Continuous("in_j", [0, np.log(150 + 1)]),
        opti.Continuous("in_k", [0, np.log(250 + 1)]),
    ],
    outputs=[
        opti.Continuous("out_a"),
        opti.Continuous("out_b"),
        opti.Continuous("out_c"),
        opti.Continuous("out_d"),
        opti.Continuous("out_e"),
    ],
    objectives=[
        opti.Maximize("out_a"),
        opti.Maximize("out_b"),
        opti.Maximize("out_c"),
        opti.Maximize("out_d"),
        opti.Minimize("out_e"),
    ],
    constraints=[
        opti.LinearEquality(["in_a", "in_b", "in_c"], rhs=1),
        #        opti.LinearEquality(["in_b", "in_c"], rhs=1),
        #        opti.LinearEquality(["in_a"], rhs=0)
    ],
)


print(problem)
_problem = reduce(problem)
print(_problem)
"""

# def f(X: pd.DataFrame) -> pd.DataFrame:
#    Y = X.iloc[:,:2].copy()
#    cols = Y.columns
#    #Y = Y.rename(columns={cols[0]:"out1", cols[1]:"out2"})
#    return Y
#
# from opti.problem import read_json
#
# problem = read_json("examples/bread.json")
# problem.f = f
# _problem = reduce(problem)
#
# cols = problem.data.columns
#
# print(_problem.data)
# print(ReducedProblem.augment_data(_problem.data, _problem._equalities, names=cols))
# print(problem.data)
