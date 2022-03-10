import numpy as np
import pytest

import opti
from opti.parameter import Continuous, Discrete


def sample_and_check_function(problem):
    X = problem.sample_inputs(10)
    X.index += 1000  # change index to test wether it is kept
    Y = problem.f(X)
    assert np.all(X.index == Y.index)
    assert np.all(problem.outputs.contains(Y))


def test_single_objective_problems():
    for _Problem in (
        opti.problems.Ackley,
        opti.problems.Branin,
        opti.problems.Himmelblau,
        opti.problems.Michalewicz,
        opti.problems.Rastrigin,
        opti.problems.Rosenbrock,
        opti.problems.Schwefel,
        opti.problems.Sphere,
        opti.problems.ThreeHumpCamel,
        opti.problems.Zakharov,
        opti.problems.Zakharov_Categorical,
        opti.problems.Zakharov_Constrained,
        opti.problems.Zakharov_NChooseKConstraint,
    ):
        problem = _Problem()
        sample_and_check_function(problem)
        optima = problem.get_optima()
        px = optima[problem.inputs.names]
        py = optima[problem.outputs.names]
        assert np.allclose(problem.f(px), py, atol=1e-04)


def test_multi_objective_problems():
    for _Problem in (
        opti.problems.Daechert1,
        opti.problems.Daechert2,
        opti.problems.Daechert3,
        opti.problems.Hyperellipsoid,
        opti.problems.OmniTest,
        opti.problems.Poloni,
        opti.problems.Qapi1,
        opti.problems.WeldedBeam,
    ):
        problem = _Problem()
        sample_and_check_function(problem)


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
    sample_and_check_function(problem)

    optima = problem.get_optima(10)
    optima.loc[0, "y1"] = -1000
    optima.loc[0, "y2"] = -1


def test_detergent():
    problem = opti.problems.Detergent()
    sample_and_check_function(problem)

    problem = opti.problems.Detergent_OutputConstraint(discrete=False)
    assert isinstance(problem.outputs["stable"], Continuous)
    sample_and_check_function(problem)

    problem = opti.problems.Detergent_OutputConstraint(discrete=True)
    assert isinstance(problem.outputs["stable"], Discrete)
    sample_and_check_function(problem)

    problem = opti.problems.Detergent_TwoOutputConstraints()
    X = problem.sample_inputs(1000)
    Y = problem.f(X)
    # test that y1 - y5 is only available if stability 1 = 1
    cols = [f"y{i+1}" for i in range(5)]
    assert np.all(Y[cols][Y["stability 1"] == 0].isna())
    assert np.all(Y[cols][Y["stability 1"] == 1].notna())

    problem = opti.problems.Detergent_NChooseKConstraint()
    # problem.create_initial_data(10)  # sampling for n-choose-k constraints not implemented


def test_zdt_problems():
    for _Problem in (
        opti.problems.ZDT1,
        opti.problems.ZDT2,
        opti.problems.ZDT3,
        opti.problems.ZDT4,
        opti.problems.ZDT6,
    ):
        problem = _Problem(n_inputs=10)
        sample_and_check_function(problem)
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
        sample_and_check_function(problem)


def test_mixed_variables_problems():
    problem = opti.problems.DiscreteVLMOP2(5)
    x = problem.inputs.sample(1000)
    y = problem.f(x)
    assert np.all(problem.outputs.contains(y))

    problem = opti.problems.DiscreteFuelInjector()
    x = problem.inputs.sample(1000)
    y = problem.f(x)
    assert np.all(problem.outputs.contains(y))
