import numpy as np
import pandas as pd
import pytest
from scipy import stats

import opti
from opti import Continuous, Problem
from opti.constraint import LinearEquality
from opti.objective import CloseToTarget, Maximize, Minimize
from opti.parameter import Discrete
from opti.problems.noisify import _add_noise_to_data, noisify_problem_with_gaussian


def check_function(problem):
    X = problem.sample_inputs(10)
    X.index += 1000  # change index to test wether it is kept
    Y = problem.f(X)
    assert np.all(X.index == Y.index)
    assert np.all(problem.outputs.contains(Y))


def test_single_objective_problems():
    for _Problem in (
        opti.problems.Ackley,
        opti.problems.Himmelblau,
        opti.problems.Rastrigin,
        opti.problems.Rosenbrock,
        opti.problems.Schwefel,
        opti.problems.Sphere,
        opti.problems.Zakharov,
        opti.problems.Zakharov_Categorical,
        opti.problems.Zakharov_Constrained,
        opti.problems.Zakharov_NChooseKConstraint,
    ):
        problem = _Problem()
        check_function(problem)
        optima = problem.get_optima()
        px = optima[problem.inputs.names]
        py = optima[problem.outputs.names]
        assert np.allclose(problem.f(px), py, atol=1e-04)


def test_multi_objective_problems():
    for _Problem in (
        opti.problems.Qapi1,
        opti.problems.Hyperellipsoid,
        opti.problems.Daechert1,
        opti.problems.Daechert2,
        opti.problems.Daechert3,
    ):
        problem = _Problem()
        check_function(problem)


def test_dataset_problems():
    for _Problem in (
        opti.problems.Alkox,
        opti.problems.BaumgartnerAniline,
        opti.problems.BaumgartnerBenzamide,
        opti.problems.Benzylation,
        opti.problems.Cake,
        opti.problems.Fullerenes,
        opti.problems.HPLC,
        opti.problems.Photodegradation,
        opti.problems.ReizmanSuzuki,
        opti.problems.Suzuki,
        opti.problems.SnAr,
    ):
        problem = _Problem()
        data = problem.get_data()
        assert problem.inputs.contains(data).all()
        # assert problem.outputs.contains(data).all()


def test_hyperellipsoid_problem():
    # Hypersphere
    problem = opti.problems.Hyperellipsoid(n=3)
    optima = problem.get_optima(10)
    assert len(optima) == 10

    # Hyperellipsoid
    with pytest.raises(ValueError):
        opti.problems.Hyperellipsoid(n=3, a=[1, 1])  # a has wrong shape

    problem = opti.problems.Hyperellipsoid(n=2, a=[1000, 1])
    check_function(problem)

    optima = problem.get_optima(10)
    optima.loc[0, "y1"] = -1000
    optima.loc[0, "y2"] = -1


def test_detergent():
    problem = opti.problems.Detergent()
    check_function(problem)

    problem = opti.problems.Detergent_OutputConstraint()
    assert isinstance(problem.outputs["stable"], Continuous)
    check_function(problem)

    problem = opti.problems.Detergent_OutputConstraint(discrete=True)
    assert isinstance(problem.outputs["stable"], Discrete)
    check_function(problem)

    problem = opti.problems.Detergent_NChooseKConstraint()
    # # problem.create_initial_data(10)  # sampling for n-choose-k constraints not implemented


def test_zdt_problems():
    for _Problem in (
        opti.problems.ZDT1,
        opti.problems.ZDT2,
        opti.problems.ZDT3,
        opti.problems.ZDT4,
        opti.problems.ZDT6,
    ):
        problem = _Problem(n_inputs=10)
        check_function(problem)
        optima = problem.get_optima(50)
        assert len(optima) == 50


def test_univariate_problems():
    for _Problem in (
        opti.problems.Line1D,
        opti.problems.Parabola1D,
        opti.problems.Sinus1D,
        opti.problems.Sigmoid1D,
        opti.problems.Step1D,
    ):
        problem = _Problem()
        check_function(problem)


def test_mixed_variables_problems():
    problem = opti.problems.DiscreteVLMOP2(5)
    x = problem.inputs.sample(1000)
    y = problem.f(x)
    assert np.all(problem.outputs.contains(y))

    problem = opti.problems.DiscreteFuelInjector()
    x = problem.inputs.sample(1000)
    y = problem.f(x)
    assert np.all(problem.outputs.contains(y))


def test_add_noise_to_data():
    cake = opti.problems.Cake()
    cake.f = lambda df: pd.DataFrame(
        data=np.zeros((len(df), cake.n_outputs)),
        columns=cake.outputs.names,
    )

    def no_noise(Y):
        return Y

    def to_zero(Y):
        return Y * 0

    Y = cake.get_data()[cake.outputs.names]
    Y_not_noisy = _add_noise_to_data(Y, [no_noise] * len(Y), cake.outputs)
    assert np.allclose(Y, Y_not_noisy)

    # outputs are clipped to their domain bounds after noise is applied, which is why the first output ("calories") is set to 300.
    Y_not_noisy = _add_noise_to_data(Y, [to_zero] * len(Y), cake.outputs)
    assert np.allclose(Y_not_noisy, [300, 0, 0])


def test_noisify_problem_with_gaussian():
    n_samples = 5000
    mu = 0.1
    sigma = 0.05

    zdt2 = opti.problems.ZDT2()
    zdt2.create_initial_data(n_samples)
    zdt2_gaussian = noisify_problem_with_gaussian(zdt2, mu=mu, sigma=sigma)

    reference = np.clip(stats.norm.rvs(loc=mu, scale=sigma, size=n_samples), 0, 1)
    Y_noise = zdt2_gaussian.get_Y() - zdt2.get_Y()
    for col in Y_noise.T:
        s, _ = stats.ks_2samp(col, reference)
        assert s < 0.1


def test_sanitize_problem():
    def _test(original):
        sanitized = opti.problems.sanitize_problem(original)

        # check inputs
        for p1, p2 in zip(original.inputs, sanitized.inputs):
            assert type(p1) == type(p2)
            if isinstance(p2, Continuous):
                p2.bounds == (0, 1)  # continuous input bounds are [0, 1]
            assert p2.name.startswith("input")

        # check outputs
        for p1, p2 in zip(original.outputs, sanitized.outputs):
            assert type(p1) == type(p2)
            if isinstance(
                p2, Continuous
            ):  # continuous output bounds are [0, 1] if specified and [-inf, inf] if not
                if np.isneginf(p1.low):
                    assert np.isneginf(p2.low)
                else:
                    assert np.isclose(p2.low, 0)
                if np.isinf(p1.high):
                    assert np.isinf(p2.high)
                else:
                    assert np.isclose(p2.high, 1)
            assert p2.name.startswith("output")

        # check objectives
        for (p1, p2) in zip(original.objectives, sanitized.objectives):
            assert type(p1) == type(p2)
            assert p2.name.startswith("output")
            assert np.isfinite(p2.target)

        # check constraints
        if original.constraints is not None:
            for c1, c2 in zip(original.constraints, sanitized.constraints):
                # sanitizing should not modify the original problem
                assert c1.names != c2.names

        # check data
        assert original.data.shape == sanitized.data.shape
        assert set(sanitized.data.columns) == set(
            sanitized.inputs.names + sanitized.outputs.names
        )
        assert sanitized.inputs.contains(sanitized.data[sanitized.inputs.names]).all()
        assert sanitized.constraints.satisfied(sanitized.data).all()

        # problem.f is dropped
        assert not hasattr(sanitized, "f")

    _test(opti.problems.Cake())

    _test(opti.problems.Photodegradation())

    z = opti.problems.Zakharov_Constrained()
    z.create_initial_data(n_samples=1000)
    _test(z)

    tp = Problem(
        inputs=[
            Continuous("secret_ingredient_1", domain=[0, 150]),
            Continuous("secret_ingredient_2", domain=[0, 50]),
            Continuous("secret_ingredient_3", domain=[0, 34.3]),
        ],
        outputs=[
            Continuous("ultimate_goal", domain=[0, np.linalg.norm([150, 50, 34.3])]),
            Continuous("mega_goal", domain=[0, np.sum(np.square([150, 50, 34.3]))]),
            Continuous(
                "super_goal", domain=[0, np.linalg.norm([150, 50, 34.3]) ** 0.5]
            ),
        ],
        objectives=[
            CloseToTarget("ultimate_goal", target=100),
            Minimize("mega_goal", target=1000),
            Maximize("super_goal", target=10),
        ],
        constraints=[
            LinearEquality(
                ["secret_ingredient_1", "secret_ingredient_2", "secret_ingredient_3"],
                lhs=np.array([3.5, 4.5, 2]),
                rhs=153.1,
            )
        ],
        f=lambda df: pd.DataFrame(
            {
                "ultimate_goal": np.linalg.norm(df.to_numpy(), axis=1),
                "mega_goal": np.sum(df ** 2, axis=1),
                "super_goal": np.linalg.norm(df, axis=1) ** 0.5,
            }
        ),
    )
    tp.create_initial_data(n_samples=100)
    _test(tp)

    empty = Problem([], [])  # this should not be possible
    with pytest.raises(TypeError):
        opti.problems.sanitize_problem(empty)
