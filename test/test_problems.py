import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import opti
from opti import Categorical, Continuous, Discrete, Problem
from opti.problems.multi_fidelity import create_multi_fidelity
from opti.problems.noisify import (
    NoiseType,
    _add_noise_to_data,
    noisify_problem,
    noisify_problem_with_gaussian,
    noisify_problem_with_scipy_stats,
)
from opti.sampling import constrained_sampling


def test_single_objective_problems():
    for problem in (
        opti.problems.single.Ackley(),
        opti.problems.single.Himmelblau(),
        opti.problems.single.Rastrigin(),
        opti.problems.single.Rosenbrock(),
        opti.problems.single.Schwefel(),
        opti.problems.single.Sphere(),
        opti.problems.single.Zakharov(),
        opti.problems.single.Zakharov_Categorical(),
        opti.problems.single.Zakharov_Constrained(),
        opti.problems.single.Zakharov_NChooseKConstraint(),
    ):
        # test json-serializability
        json.dumps(problem.to_config())

        # test function evaluations
        x = problem.sample_inputs(10)
        y1 = problem.f(x.values)
        y2 = problem.eval(x)
        assert np.allclose(y1, y2["y"])
        assert np.all(problem.outputs.contains(y2))

        # test the optima
        optima = problem.get_optima()
        px = optima[problem.inputs.names]
        py = optima[problem.outputs.names]
        assert np.allclose(problem.eval(px), py, atol=1e-04)


def test_multi_objective_problems():
    for problem in (
        opti.problems.Qapi1(),
        opti.problems.Hyperellipsoid(),
        opti.problems.Daechert1(),
        opti.problems.Daechert2(),
        opti.problems.Daechert3(),
        opti.problems.Detergent(),
    ):
        json.dumps(problem.to_config())  # test serializing

        # evaluate the function
        x = problem.inputs.sample(10)
        y1 = problem.f(x.values)
        y2 = problem.eval(x)
        assert np.allclose(y1, y2[problem.outputs.names].values)
        assert np.all(problem.outputs.contains(y2))

        # evaluate the constraints
        if problem.constraints:
            problem.constraints.eval(x)
            problem.constraints.satisfied(x)


def test_baking_problems():
    problem = opti.problems.Cake()
    json.dumps(problem.to_config())
    assert set(problem.inputs.names + problem.outputs.names) == set(
        problem.data.columns
    )

    problem = opti.problems.Bread()
    json.dumps(problem.to_config())


def test_dataset_problems():
    for _Problem in (
        opti.problems.Alkox,
        opti.problems.BaumgartnerAniline,
        opti.problems.BaumgartnerBenzamide,
        opti.problems.Benzylation,
        opti.problems.Fullerenes,
        opti.problems.HPLC,
        opti.problems.Photodegradation,
        opti.problems.ReizmanSuzuki,
        opti.problems.Suzuki,
        opti.problems.SnAr,
    ):
        problem = _Problem()
        assert problem.inputs.contains(problem.data[problem.inputs.names]).all()


def test_hyperellipsoid_problem():
    # Hypersphere
    problem = opti.problems.Hyperellipsoid(n=3)
    json.dumps(problem.to_config())
    x = problem.inputs.sample()
    y = problem.eval(x)
    assert problem.outputs.contains(y)

    # sampling from the constrained input space should work
    problem.create_initial_data(10)
    assert len(problem.data) == 10

    # sampling from the Pareto front
    front = problem.get_optima(1)
    assert len(front) == 1
    front = problem.get_optima(10)
    assert len(front) == 10

    # Hyperellipsoid
    with pytest.raises(ValueError):
        opti.problems.Hyperellipsoid(n=3, a=[1, 1])  # a has wrong shape

    problem = opti.problems.Hyperellipsoid(n=2, a=[1000, 1])
    x = problem.inputs.sample()
    y = problem.eval(x)

    front = problem.get_optima(10)
    front.loc[0, "y0"] = -1000
    front.loc[0, "y1"] = -1


def test_detergent():
    p1 = opti.problems.Detergent()
    p1.create_initial_data(1)
    p1.create_initial_data(10)

    p2 = opti.problems.Detergent_OutputConstraint()
    p2.create_initial_data(1)
    p2.create_initial_data(10)

    p2 = opti.problems.Detergent_NChooseKConstraint()
    # p2.create_initial_data(1)  # sampling for n-choose-k constraints not implemented
    # p2.create_initial_data(10)


def test_zdt_problems():
    for problem in (
        opti.problems.ZDT1(n_inputs=20),
        opti.problems.ZDT2(n_inputs=20),
        opti.problems.ZDT3(n_inputs=20),
        opti.problems.ZDT4(n_inputs=20),
        opti.problems.ZDT6(n_inputs=20),
    ):
        json.dumps(problem.to_config())

        x = problem.inputs.sample()
        y = problem.eval(x)
        assert np.all(problem.outputs.contains(y))

        # test obtaining the true pareto front
        opt = problem.get_optima(points=50)
        assert len(opt) == 50


def test_univariate_problems():
    for problem in (
        opti.problems.Line1D(),
        opti.problems.Parabola1D(),
        opti.problems.Sinus1D(),
        opti.problems.Sigmoid1D(),
        opti.problems.Step1D(),
    ):
        x = problem.inputs.sample()
        y = problem.eval(x)
        assert np.all(problem.outputs.contains(y))


def test_mixed_variables_problems():
    problem = opti.problems.DiscreteVLMOP2(5)
    x = problem.inputs.sample(1000)
    y = problem.eval(x)
    assert np.all(problem.outputs.contains(y))

    problem = opti.problems.DiscreteFuelInjector()
    x = problem.inputs.sample(1000)
    y = problem.eval(x)
    assert np.all(problem.outputs.contains(y))


def test_add_noise_to_data():
    cake = opti.problems.Cake()
    cake.f = lambda x: np.zeros((len(x), cake.n_outputs))
    Y = cake.get_Y()

    def noisify(Y):
        return Y

    Y_not_noisy = _add_noise_to_data(Y, [noisify] * Y.shape[1], cake.outputs)
    assert np.allclose(Y, Y_not_noisy)

    def noisify(Y):
        return Y * 0

    # outputs are clipped to their domain bounds after noise is applied
    # this is why the first output ("calories") is set to 300
    Y_not_noisy = _add_noise_to_data(Y, [noisify] * Y.shape[1], cake.outputs)
    assert np.allclose(Y_not_noisy, [300, 0, 0])


def test_noisify_problem():
    cake = opti.problems.Cake()
    cake.outputs["calories"].domain = [0, 600]
    cake.outputs["calories"].low = 0

    def test(rvs, rv_kwargs):
        for noise_type, neutral_element in (
            (NoiseType.ADDITIVE, 0),
            (NoiseType.MULTIPLICATIVE, 1),
            (None, 0),
        ):
            cake.f = lambda x: np.full((len(x), cake.n_outputs), neutral_element)

            # create a large enough data set for a KS test
            X = cake.inputs.sample(1000)
            Y = pd.DataFrame(
                columns=cake.outputs.names,
                data=np.full((1000, cake.n_outputs), neutral_element),
            )
            cake.data = pd.concat([X, Y], axis=1)

            if noise_type is None:
                noisy_cake = noisify_problem_with_scipy_stats(
                    cake, rvs, rv_kwargs=rv_kwargs
                )
            else:
                noisy_cake = noisify_problem_with_scipy_stats(
                    cake, rvs, rv_kwargs=rv_kwargs, noise_types=noise_type
                )

            # test that both data and function are noisified
            noisy_Y = np.row_stack([noisy_cake.get_Y(), noisy_cake.f(X.values)])

            rv_kwargs_ref = rv_kwargs if isinstance(rv_kwargs, list) else [rv_kwargs]
            rvs_ref = rvs if isinstance(rvs, list) else [rvs]
            for col, rv, rvk in zip(noisy_Y.T, rvs_ref, rv_kwargs_ref):
                reference = rv.rvs(**rvk, size=col.size)
                # since we clip noisy problems to the bounds we also clip the reference
                reference[reference < 0] = 0
                s, _ = stats.ks_2samp(col, reference)
                assert s < 0.1

    test(stats.truncnorm, {"a": 0.1, "b": 2})
    test(
        [stats.truncnorm, stats.norm, stats.norm],
        [{"a": 0.1, "b": 2}, {"loc": 0.1, "scale": 0.3}, {"loc": -0.1, "scale": 0.1}],
    )


def test_noisify_categorical_discrete_outputs():
    problem = Problem(
        inputs=[Continuous("i1", [0, 1])],
        outputs=[
            Categorical("o1", [1, 2, 7]),
            Discrete("o2", [0, 1]),
            Discrete("o3", [0, 1, 2]),
        ],
        f=lambda x: np.array([[1, 1, 2]]),
    )

    def noisify(Y):
        return Y + 1

    noisy_cat_disc_problem = noisify_problem(problem, noisify)

    # Cat. dummy value will be increased by 1, discrete will be out of bounds after + 1
    assert np.allclose(noisy_cat_disc_problem.f(1), np.array([[2, 1, 2]]))


def test_noisify_problem_with_gaussian():
    n_samples = 5000
    mu = 0.1
    sigma = 0.05
    reference = np.clip(stats.norm.rvs(loc=mu, scale=sigma, size=n_samples), 0, 1)

    zdt2 = opti.problems.ZDT2()
    zdt2.create_initial_data(n_samples)
    zdt2_gaussian = noisify_problem_with_gaussian(zdt2, mu=mu, sigma=sigma)
    Y_gaussian = zdt2_gaussian.get_Y()
    Y = zdt2.get_Y()
    Y_noise = Y_gaussian - Y
    for col in Y_noise.T:
        s, _ = stats.ks_2samp(col, reference)
        assert s < 0.1


def test_multi_fidelity():
    problem = Problem(
        inputs=[Continuous("i", [0, 1])],
        outputs=[
            Continuous("o", [0, 1]),
        ],
        f=lambda x: np.array([[0.5]]),
    )

    def second_fidelity(_):
        return np.array([[1.0]])

    def first_predicate(counter, _):
        return counter % 3 == 0

    def second_predicate(counter, _):
        return not first_predicate(counter, _)

    mf_problem = create_multi_fidelity(
        problem, [problem.f, second_fidelity], [first_predicate, second_predicate]
    )
    for i in range(100):
        y = mf_problem.f(None)
        if i % 3 == 0:
            assert y[0, 0] == 0.5
        else:
            assert y[0, 0] == 1.0

    n_inputs = 5
    zdt1 = opti.problems.ZDT1(n_inputs=n_inputs)
    first_fidelity = noisify_problem_with_gaussian(zdt1, mu=0, sigma=0.01)
    second_fidelity = noisify_problem_with_gaussian(zdt1, mu=0.1, sigma=0.5)

    def pred_second(counter, _):
        return counter % 3 == 0

    def pred_first(counter, _):
        return not pred_second(counter, _)

    mf_zdt = create_multi_fidelity(
        zdt1, [first_fidelity.f, second_fidelity.f], [pred_first, pred_second]
    )

    n_samples = 3000

    Y_mf = -np.ones((n_samples, 2))
    Y = -np.ones((n_samples, 2))
    X = constrained_sampling(n_samples, mf_zdt.inputs, mf_zdt.constraints).values
    for i in range(n_samples):
        # needs to be in a loop so that the counter of mf_zdt is increased
        Y_mf[i, :] = mf_zdt.f(X[i].reshape(1, n_inputs))
        Y[i, :] = zdt1.f(X[i].reshape(1, n_inputs))

    f2_inds = np.array([pred_second(i, None) for i in range(n_samples)])
    f1_inds = (1 - f2_inds) > 0

    Y_1_noisy = Y_mf[f1_inds]
    Y_2_noisy = Y_mf[f2_inds]
    Y_1 = Y[f1_inds]
    Y_2 = Y[f2_inds]
    Y_1_noise = np.clip(Y_1_noisy - Y_1, 0, 1)
    Y_2_noise = np.clip(Y_2_noisy - Y_2, 0, 1)
    reference_1 = np.clip(stats.norm.rvs(loc=0, scale=0.01, size=np.sum(f2_inds)), 0, 1)
    reference_2 = np.clip(
        stats.norm.rvs(loc=0.1, scale=0.5, size=np.sum(f1_inds)), 0, 1
    )

    for Y_noise, ref in zip([Y_1_noise, Y_2_noise], [reference_1, reference_2]):
        for col in Y_noise.T:
            s, _ = stats.ks_2samp(col, ref)
            assert s < 0.1
