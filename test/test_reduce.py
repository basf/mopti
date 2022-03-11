from opti.constraint import (
    Constraint,
    Constraints,
    LinearEquality,
    LinearInequality,
    NonlinearEquality,
    NonlinearInequality,
)
from opti.tools.reduce import find_linear_equality_constraints


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


# def test_find_continuous_inputs():
# TODO


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
