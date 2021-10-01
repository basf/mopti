import numpy as np
import pytest

from opti.constraint import (
    Constraints,
    LinearEquality,
    LinearInequality,
    NChooseK,
    NonlinearEquality,
    NonlinearInequality,
)
from opti.parameter import Categorical, Continuous, Discrete, Parameters
from opti.sampling import (
    constrained_sampling,
    polytope,
    rejection_sampling,
    simplex,
    sobol_sampling,
    sphere,
)


def test_simplex_grid():
    grid = simplex.grid(dimension=3, levels=5)
    assert grid.shape[1] == 3
    assert (grid.sum(axis=1) == 1).all()
    assert (np.unique(grid) == np.array([0, 0.25, 0.5, 0.75, 1])).all()


def test_simplex_sampling():
    points = simplex.sample(dimension=3, n_samples=100)
    assert points.shape == (100, 3)
    assert np.allclose(points.sum(axis=1), 1)
    assert (points >= 0).all()
    assert (points <= 1).all()


def test_sphere_sampling():
    points = sphere.sample(dimension=5, n_samples=1000)
    assert points.shape == (1000, 5)
    assert (np.abs(points) < 1).all()

    points = sphere.sample(dimension=5, n_samples=1000, positive=True)
    assert (points > 0).all()
    assert (points < 1).all()


def test_polytope_chebyshev_center():
    A = np.row_stack([-np.eye(3), np.eye(3)])
    b = np.r_[-np.zeros(3), np.ones(3)]
    x0 = polytope._chebyshev_center(A, b)
    assert np.allclose(x0, [0.5, 0.5, 0.5])


def test_polytope_affine_subspace():
    # x1 + x2 + x3 = 2
    A = np.array([[1, 1, 1]])
    b = np.array([2])
    N, xp = polytope._affine_subspace(A, b)
    Z = np.random.uniform(low=-1, high=1, size=(5, 2))
    X = Z @ N.T + xp
    assert np.allclose(X @ A.T, b)


# def test_polytope_sample():
#     # Sampling from unit-simplex in R^3
#     X = polytope.sample(
#         samples=100,
#         lower=[0, 0, 0],
#         upper=[1, 1, 1],
#         A2=np.array([[1, 1, 1]]),
#         b2=np.array([1]),
#     )
#     assert np.allclose(X.sum(axis=1), 1)
#     assert (X >= 0).all()
#     assert (X <= 1).all()

#     # Sampling from [0, 1]^2 subject to
#     # x1 / 2 + x2 <= 1
#     # 2/3 x1 - x2 <= -0.2
#     A1 = np.array([[1 / 2, 1], [2 / 3, -1]])
#     b1 = np.array([1, -0.2])
#     X = polytope.sample(samples=100, lower=[0, 0], upper=[1, 1], A1=A1, b1=b1)
#     assert np.all(X @ A1.T <= b1)


def test_polytope_sampling():
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(8)])
    constraints = Constraints(
        [
            LinearEquality(names=["x0", "x1", "x2", "x3"], lhs=1, rhs=1),
            LinearEquality(names=["x4", "x5", "x6"], lhs=1, rhs=1),
            LinearInequality(names=["x0", "x1", "x2"], lhs=1, rhs=0.3),
        ]
    )
    X = polytope.polytope_sampling(100, parameters, constraints)
    assert parameters.contains(X).all()
    assert constraints.satisfied(X).all()

    # test without equalities
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints(
        [LinearInequality(names=["x0", "x1", "x2"], lhs=1, rhs=1)]
    )
    X = polytope.polytope_sampling(100, parameters, constraints)
    assert parameters.contains(X).all()
    assert constraints.satisfied(X).all()

    # test without inequalities
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints([LinearEquality(names=["x0", "x1", "x2"], lhs=1, rhs=1)])
    X = polytope.polytope_sampling(100, parameters, constraints)
    assert parameters.contains(X).all()
    assert constraints.satisfied(X).all()


def test_rejection_sampling():
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])

    # test sampling from unit-simplex
    constraints = Constraints(
        [LinearInequality(names=["x0", "x1", "x2"], lhs=1, rhs=1)]
    )
    X = rejection_sampling(100, parameters, constraints)
    assert parameters.contains(X).all()
    assert constraints.satisfied(X).all()

    # test sampling from unit-sphere
    constraints = Constraints([NonlinearInequality("x0**2 + x1**2 + x2**2 - 1")])
    X = rejection_sampling(100, parameters, constraints)
    assert parameters.contains(X).all()
    assert constraints.satisfied(X).all()

    # test max_iters exceeded
    with pytest.raises(Exception):
        rejection_sampling(10000, parameters, constraints, max_iters=1)


def test_constrained_sampling():
    # linear equalities -> polytope sampling
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints([LinearEquality(names=["x0", "x1", "x2"], lhs=1, rhs=1)])
    X = constrained_sampling(10, parameters, constraints)
    assert X.shape == (10, 3)

    # nonlinear inequalities -> rejection sampling
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints([NonlinearInequality("x0**2 + x1**2 + x2**2 - 1")])
    X = constrained_sampling(10, parameters, constraints)
    assert X.shape == (10, 3)

    # non-linear equalities -> error
    parameters = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints([NonlinearEquality("x0**2 + x1**2 + x2**2 - 1")])
    with pytest.raises(Exception):
        constrained_sampling(10, parameters, constraints)

    # linear equalities and non-continuous parameters -> error
    parameters = Parameters(
        [Continuous("x0", domain=[0, 1]), Discrete("x1", [0, 1, 2])]
    )
    constraints = Constraints([NonlinearEquality("x0**2 + x1**2 - 1")])
    with pytest.raises(Exception):
        constrained_sampling(10, parameters, constraints)


def test_constrained_nchoosek():
    # two n-choose-k constraints
    parameters = Parameters([Continuous(f"x{i+1}", domain=[0, 1]) for i in range(10)])
    constraints = Constraints(
        [
            NChooseK(["x1", "x2", "x3", "x4", "x5"], max_active=3),
            NChooseK(["x6", "x7", "x8", "x9", "x10"], max_active=3),
        ]
    )
    X = constrained_sampling(10, parameters, constraints)
    assert constraints.satisfied(X).all()

    # n-choose-k constraint + a particular inequality constraint
    parameters = Parameters([Continuous(f"x{i+1}", domain=[0, 1]) for i in range(3)])
    constraints = Constraints(
        [
            NChooseK(["x1", "x2", "x3"], max_active=2),
            LinearInequality(["x1", "x2", "x3"], lhs=1, rhs=1),
        ]
    )
    X = constrained_sampling(10, parameters, constraints)
    assert constraints.satisfied(X).all()


def test_sobol_sampling():
    parameters = Parameters(
        [
            Continuous("x1", domain=[10, 20]),
            Discrete("x2", domain=[1, 9, 27, 28]),
            Categorical("x3", domain=["A", "B", "C"]),
        ]
    )
    X = sobol_sampling(100, parameters)
    assert np.all(parameters.contains(X))
