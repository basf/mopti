import numpy as np

import opti
from opti.metric import pareto_front


def test_pareto_efficient():
    # case 1:
    x = np.linspace(0, 10, 1000)
    A = np.zeros((1000, 2))
    A[:, 0] = (x - 2) ** 2
    A[:, 1] = 3 * x

    efficient = opti.metric.is_pareto_efficient(A)

    # check pareto-efficient points
    for a in A[efficient]:
        # only the point itself is allowed to be strictly equal or better on all objectives
        assert np.all(A <= a, axis=1).sum() == 1
        # all other pareto points must be equal or better in at least one objective
        assert np.any(A[efficient] <= a, axis=1).sum() == efficient.sum()

    # check dominated points
    for a in A[~efficient]:
        # point cannot be better than any pareto optimal point in any objective
        assert np.any(A[efficient] <= a, axis=1).any()

    # case 2:
    A = np.array(
        [
            [0.2, 0.6],
            [0.1, 0.7],
            [0.0, 1.3],
            [0.5, 0.3],
            [0.0, 3.0],
            [0.0, 3.1],
            [5.0, 5.0],
        ]
    )
    efficient = opti.metric.is_pareto_efficient(A)
    assert len(A[efficient]) == 4
    assert np.alltrue(pareto_front(A) == A[efficient])


def test_crowding_distance():
    # concrete example
    # - for the two outer points the crowding distance should be inf (missing neighbor)
    # - for the inner two points 0.7 is the average cuboide side length and 0.9 is the
    #   scaling from the spread in each dimension
    A = np.array([[0.1, 1], [0.3, 0.8], [0.8, 0.3], [1, 0.1]])
    np.testing.assert_array_almost_equal(
        opti.metric.crowding_distance(A),
        np.array([np.inf, 0.7 / 0.9, 0.7 / 0.9, np.inf]),
    )


def test_generational_distance():
    # Test case: the reference front forms a unit quarter circle around (1, 1)
    φ = np.linspace(0, np.pi / 2, 100)
    R = 1 - np.array([np.sin(φ), np.cos(φ)]).T
    A = np.array([[0.1, 1], [0.3, 0.8], [0.8, 0.3], [1, 0.1]])

    # single point on the front
    gd = opti.metric.generational_distance(np.array([[1, 0]]), R)
    assert np.isclose(gd, 0)
    # single point at distance = 0.1 to the front
    gd = opti.metric.generational_distance(np.array([[1, 0.1]]), R)
    assert np.isclose(gd, 0.1)
    # multiple points
    gd = opti.metric.generational_distance(A, R)
    assert np.isclose(gd, 0.18603015677)


def test_inverted_generational_distance():
    # Test case: the reference front forms a unit quarter circle around (1, 1)
    φ = np.linspace(0, np.pi / 2, 100)
    R = 1 - np.array([np.sin(φ), np.cos(φ)]).T
    A = np.array([[0.1, 1], [0.3, 0.8], [0.8, 0.3], [1, 0.1]])

    # single point at distance = 1 to the front
    igd = opti.metric.inverted_generational_distance(np.array([[1, 1]]), R)
    assert np.isclose(igd, 1)
    # multiple points
    igd = opti.metric.inverted_generational_distance(A, R)
    assert np.isclose(igd, 0.292415441318418)
