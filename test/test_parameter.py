import json
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from opti.parameter import Categorical, Continuous, Discrete, Parameters, make_parameter


def test_arbitrary_parameters():
    # Issue #60
    # saves as {}} when no extra fields
    p = make_parameter(name="foo", type="continuous", domain=[0, 1])
    assert len(p.extra_fields) == 0

    # saves extra fields as a dictionary, of expected form. Independent of type.
    conf_conf = []
    for key, value in {
        "continuous": [0, 1],
        "discrete": [1, 2, 5],
        "categorical": ["A", "B", 3],
    }.items():
        e_dict = {"extra": True, "info": [1, 2, 3], "here": {"a": 1, "b": 2}}
        conf = {"name": "foo" + key, "type": key, "domain": value}
        conf.update(e_dict)
        conf_conf.append(conf)
        p = make_parameter(
            name="foo" + key,
            type=key,
            domain=value,
            extra=True,
            info=[1, 2, 3],
            here={"a": 1, "b": 2},
        )
        TestCase().assertDictEqual(p.extra_fields, e_dict)
        TestCase().assertDictEqual(p.to_config(), conf)

    # test that the parameter constructors and to_config are inverses
    [
        TestCase().assertDictEqual(Parameters(conf_conf).to_config()[i], conf_conf[i])
        for i in range(len(conf_conf))
    ]
    # saves extra fields as a dictionary, of expected form


def test_make_parameter():
    p = make_parameter(name="foo", type="continuous", domain=[0, 1])
    assert isinstance(p, Continuous)
    p = make_parameter(name="foo", type="discrete", domain=[1, 2, 5])
    assert isinstance(p, Discrete)
    p = make_parameter(name="foo", type="categorical", domain=["A", "B", 3])
    assert isinstance(p, Categorical)

    # no bounds
    p = make_parameter(name="foo", type="continuous")
    assert isinstance(p, Continuous)
    with pytest.raises(ValueError):
        make_parameter(name="foo", type="discrete")
    with pytest.raises(ValueError):
        make_parameter(name="foo", type="categorical")


class TestContinuous:
    def test_init(self):
        for domain in ((1, 10), [1, 10], np.array([1, 10])):
            p = Continuous("x", domain=domain)
            assert p.name == "x"
            assert p.bounds == (1, 10)
            assert p.domain == [1, 10]

    def test_checks(self):
        with pytest.raises(ValueError):
            Continuous("x", domain=[20, 1])
        with pytest.raises(ValueError):
            Continuous("x", domain=[1, 2, 3])

    def test_config(self):
        p = Continuous("x", domain=[1, 10])
        conf = p.to_config()
        assert conf["type"] == "continuous"
        assert conf["name"] == "x"
        assert conf["domain"] == [1, 10]
        json.dumps(conf)

        # json-compliance: inf -> None
        p = Continuous("x")
        conf = p.to_config()
        assert conf["domain"] == [None, None]

    def test_config_int32(self):
        # test if domain with np.int32 can be serialized
        p = Continuous("x", domain=[0, np.int32(1)])
        json.dumps(p.to_config())

    def test_with_units(self):
        p = Continuous("x", domain=[1, 10], unit="°C")
        conf = p.to_config()
        assert conf["unit"] == "°C"

    def test_round(self):
        p = Continuous(name="x", domain=[1, 10])
        # scalar
        assert p.round(0) == 1
        assert p.round(3.3) == 3.3
        assert p.round(100) == 10
        # np.array
        assert np.allclose(p.round(np.zeros(10)), 1)
        # pd.Series
        assert np.allclose(p.round(pd.Series([0, 0, 0])), 1)

    def test_contains(self):
        p = Continuous(name="x", domain=[1, 10])
        # scalar
        assert p.contains(2)
        # np.array
        assert np.all(p.contains(np.array([2.3, 3, 4, 5, 6])))
        # pd.Series
        assert np.all(p.contains(pd.Series([2.3, 3, 4, 5, 6])))

    def test_sample(self):
        p = Continuous(name="x", domain=[1, 10])
        assert np.all(p.contains(p.sample()))

        p = Continuous(name="x", domain=[-np.inf, np.inf])
        assert np.all(p.contains(p.sample()))

        p = Continuous(name="x", domain=[0, np.inf])
        assert np.all(p.contains(p.sample()))

        p = Continuous(name="x", domain=[-np.inf, 0])
        assert np.all(p.contains(p.sample()))

    def test_repr(self):
        p1 = Continuous(name="x", domain=[1, 10])
        p2 = eval(p1.__repr__())
        assert p1.to_config() == p2.to_config()

    def test_fixed(self):
        p = Continuous("x", domain=[7, 7])
        assert p.bounds == (7, 7)
        assert np.allclose(p.sample(), 7)

    def test_unbounded(self):
        p = Continuous("x", domain=[-np.inf, np.inf])
        assert p.contains(p.sample()).all()

        p = Continuous("x", domain=[None, None])
        assert p.contains(p.sample()).all()

        p = Continuous("x")
        assert p.contains(p.sample()).all()

    def test_unit_range(self):
        p = Continuous(name="x", domain=[-10, 20])
        x = np.linspace(-10, 20)
        xt = p.to_unit_range(x)
        assert np.allclose(xt, np.linspace(0, 1))

        p = Continuous(name="x", domain=[1, 1])  # fixed -> don't do anything
        x = np.linspace(-10, 20)
        xt = p.to_unit_range(x)
        assert np.allclose(xt, x)


class TestDiscrete:
    def test_init(self):
        p = Discrete(domain=[1, 2, 3, 5], name="x")
        assert p.name == "x"
        assert p.bounds == (1, 5)
        assert p.domain == [1, 2, 3, 5]

        p = Discrete(domain=[-1.1, 5.1], name="x")
        assert p.bounds == (-1.1, 5.1)
        assert p.domain == [-1.1, 5.1]

    def test_config(self):
        p = Discrete(domain=[1, 2, 3, 5], name="x")
        conf = p.to_config()
        assert conf["type"] == "discrete"
        assert conf["name"] == "x"
        assert conf["domain"] == [1, 2, 3, 5]
        json.dumps(conf)

    def test_config_int32(self):
        # test if domain with np.int32 can be serialized
        p = Discrete("x", domain=np.array([0, 1, 2], dtype=np.int32))
        json.dumps(p.to_config())

    def test_checks(self):
        with pytest.raises(ValueError):
            Discrete("x", domain="very discrete")  # domain must be a list
        with pytest.raises(ValueError):
            Discrete("x", domain=[])  # domain needs to have at least 2 values
        with pytest.raises(ValueError):
            Discrete("x", domain=[1, 1])  # domain cannot contain duplicates
        with pytest.raises(ValueError):
            Discrete("x", domain=["a", "b"])  # domain may only contain numerics

    def test_with_units(self):
        p = Discrete("x", domain=[1, 2, 7, 10], unit="°C")
        conf = p.to_config()
        assert conf["unit"] == "°C"

    def test_is_integer(self):
        assert Discrete(name="x", domain=[0, 1, 2]).is_integer()
        assert Discrete(name="x", domain=[-1, 0, 1]).is_integer()
        assert not Discrete(name="x", domain=[1, 2, 10]).is_integer()
        assert not Discrete(name="x", domain=[1, 2.1, 3]).is_integer()
        assert not Discrete(name="x", domain=[1.1, 2.1]).is_integer()

    def test_round(self):
        p = Discrete(name="x", domain=[1, 2, 3, 10])
        # scalars
        assert p.round(0) == 1
        assert p.round(2) == 2
        assert p.round(100) == 10
        # np.array
        assert np.allclose(p.round(np.zeros(10)), 1)
        # pd.Series
        assert np.allclose(p.round(pd.Series([0, 0, 0])), 1)

        # should also work for non-integer domains
        p = Discrete(domain=[1.1, 2.5, 3.141], name="x")
        assert p.round(0) == 1.1
        assert p.round(4) == 3.141

    def test_contains(self):
        p = Discrete(name="x", domain=[1, 2, 3, 10])
        # scalar
        assert p.contains(2)
        assert not p.contains(4)
        # array
        assert np.all(p.contains([1, 1, 1, 2, 2, 2]))

    def test_sample(self):
        p = Discrete(name="x", domain=[1, 2, 3, 5])
        assert np.all(p.contains(p.sample()))

    def test_repr(self):
        p1 = Discrete(name="x", domain=[1, 2, 3, 5])
        p2 = eval(p1.__repr__())
        assert p1.to_config() == p2.to_config()

    def test_fixed(self):
        p = Discrete("x", domain=[7])
        assert p.bounds == (7, 7)
        assert np.allclose(p.sample(), 7)

    def test_unit_range(self):
        p = Discrete(name="x", domain=[-10, 5, 20])
        x = np.linspace(-10, 20)
        xt = p.to_unit_range(x)
        assert np.allclose(xt, np.linspace(0, 1))

        p = Discrete(name="x", domain=[1])  # fixed -> don't do anything
        x = np.linspace(-10, 20)
        xt = p.to_unit_range(x)
        assert np.allclose(xt, x)


class TestCategorical:
    def test_init(self):
        p = Categorical(domain=["A", "B", 3], name="x")
        assert p.name == "x"
        assert p.domain == ["A", "B", 3]

    def test_checks(self):
        with pytest.raises(ValueError):
            Categorical("x", domain="very categorical")  # domain must be a list
        with pytest.raises(ValueError):
            Categorical("x", domain=[])  # domain needs to have at least 2 values
        with pytest.raises(ValueError):
            Categorical("x", domain=["a"])  # domain needs to have 2+ values
        with pytest.raises(ValueError):
            Categorical("x", domain=["a", "a"])  # domain cannot contain duplicates

    def test_config(self):
        p = Categorical(domain=["A", "B", 3], name="x")
        conf = p.to_config()
        assert conf["type"] == "categorical"
        assert conf["name"] == "x"
        assert conf["domain"] == ["A", "B", 3]
        json.dumps(conf)

    def test_contains(self):
        p = Categorical(name="x", domain=["A", "B", 3])
        assert p.contains("A")
        assert p.contains(["A", "A", 3]).all()
        assert not p.contains("x")
        assert not p.contains(["A", "x"]).all()

    def test_sample(self):
        p = Categorical(name="x", domain=["A", "B", 3])
        x = p.sample()
        assert p.contains(x)

    def test_repr(self):
        p1 = Categorical(name="x", domain=["A", "B", 3])
        p2 = eval(p1.__repr__())
        assert p1.to_config() == p2.to_config()

    def test_round(self):
        p = Categorical(name="x", domain=["A", "B", 3])
        assert p.round("A") == "A"
        x = ["A", "A", "B", 3]
        assert np.all(p.round(x) == x)

    def test_onehot_encoding(self):
        p = Categorical(name="x", domain=["B", "A", "C"])
        points = pd.Series(["A", "A", "C", "B"])
        transformed = p.to_onehot_encoding(points)
        assert np.allclose(transformed["x§A"], [1, 1, 0, 0])
        assert np.allclose(transformed["x§B"], [0, 0, 0, 1])
        assert np.allclose(transformed["x§C"], [0, 0, 1, 0])

        untransformed = p.from_onehot_encoding(transformed)
        assert np.all(points == untransformed)

    def test_label_encoding(self):
        p = Categorical(name="x", domain=["B", "A", "C"])
        points = pd.Series(["A", "A", "C", "B"])
        transformed = p.to_label_encoding(points)
        assert np.allclose(transformed, [1, 1, 2, 0])

        untransformed = p.from_label_encoding(transformed)
        assert np.all(points == untransformed)


class TestParameters:
    mixed_parameters = Parameters(
        [
            Categorical(name="category", domain=["city bike", "race bike"]),
            Continuous(name="weight", domain=[5, 20]),
            Discrete(name="gears", domain=[3, 8, 16, 21, 24, 27]),
        ]
    )

    def test_init(self):
        # init from a list/tuple of dimensions
        Parameters(
            (
                Continuous(name="foo", domain=[1, 10]),
                Discrete(name="bar", domain=[1, 2, 3, 4]),
                Categorical(name="baz", domain=["A", "B", 3]),
            )
        )

        # init from a list/tuple of configs
        Parameters(
            [
                {"name": "foo", "type": "continuous", "domain": [1, 10]},
                {"name": "bar", "type": "discrete", "domain": [1, 2, 3, 4]},
                {"name": "baz", "type": "categorical", "domain": ["A", "B", 3]},
            ]
        )

        # not a list/tuple
        with pytest.raises(TypeError):
            Parameters(Continuous(name="weight", domain=[5, 20]))

    def test_config(self):
        # test serializing to config
        conf = self.mixed_parameters.to_config()
        params = Parameters(conf)
        assert params.to_config() == conf

    def test_repr(self):
        # repr uniquely identifies the space
        params = eval(self.mixed_parameters.__repr__())
        assert params.to_config() == self.mixed_parameters.to_config()

    def test_iter(self):
        # test iterating over the dimensions of the parameter space
        n = len([d for d in self.mixed_parameters])
        assert len(self.mixed_parameters) == n

    def test_subscriptable(self):
        # test accessing the dimensions by name
        for name in self.mixed_parameters.names:
            assert self.mixed_parameters[name].name == name

    def test_contains(self):
        # test if points are correctly identified as contained inside the parameter space
        points = pd.DataFrame(
            columns=self.mixed_parameters.names,
            data=[["race bike", 9.12, 8], ["city bike", 16.8, 8]],
        )
        assert self.mixed_parameters.contains(points).all()

    def test_sample(self):
        # sample some points from the input space and check if they are inside the bounds
        points = self.mixed_parameters.sample(3)
        assert self.mixed_parameters.contains(points).all()
        assert points.shape == (3, len(self.mixed_parameters))

        params = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(5)])
        points = params.sample(3)
        assert set(points.columns) == set(["x0", "x1", "x2", "x3", "x4"])
        assert points.shape == (3, len(params))
        assert params.contains(points).all()

        points = params.sample(1)
        assert points.shape == (1, len(params))
        assert params.contains(points).all()

    def test_round(self):
        # test rounding values
        points = pd.DataFrame(
            {
                "category": ["city bike", "race bike", "race bike", "race bike"],
                "weight": [4, 7, 100, 25],
                "gears": [1, 5, 18, 21],
            },
            index=[4, 5, 6, 7],
        )
        rounded = self.mixed_parameters.round(points)
        assert self.mixed_parameters.contains(rounded).all()

    def test_bounds(self):
        # test parameter bounds
        params = Parameters([Continuous(f"x{i}", domain=[0, 1]) for i in range(5)])
        assert (params.bounds.loc["min"] == 0).all()
        assert (params.bounds.loc["max"] == 1).all()

        params = Parameters([Discrete(f"x{i}", domain=[0, 1]) for i in range(5)])
        assert (params.bounds.loc["min"] == 0).all()
        assert (params.bounds.loc["max"] == 1).all()

        # no bounds when space includes categorical parameters
        with pytest.raises(TypeError):
            params = Parameters(self.mixed_parameters)
            params.bounds

    def test_transform(self):
        # transform to the unit-range
        params = Parameters([Continuous(f"x{i}", domain=[-10, 10]) for i in range(5)])
        X = params.sample(20)
        Xt = params.transform(X, continuous="normalize")
        assert np.all(Xt >= 0) and np.all(Xt <= 1)

        # transform with mixed parameters
        params = Parameters(
            [
                Continuous("x1", domain=[-10, 10]),
                Discrete("x2", domain=[0, 1, 5]),
                Categorical("x3", domain=["A", "B", "C"]),
            ]
        )
        X = params.sample(20)
        Xt = params.transform(X, continuous="normalize", discrete="normalize")
        assert Xt.shape == (20, 5)
        assert np.all(Xt >= 0) and np.all(Xt <= 1)

    def test_duplicate_names(self):
        # test handling of duplicate parameter names
        with pytest.raises(ValueError):
            Parameters([Continuous("x"), Continuous("x")])

    def test_add(self):
        params1 = Parameters([Continuous(f"x{i}") for i in range(3)])
        params2 = Parameters([Continuous(f"y{i}") for i in range(3)])
        both = params1 + params2
        assert isinstance(both, Parameters)
        assert len(both) == 6
