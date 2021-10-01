import json

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression

from opti.model import LinearModel, Model, Models, make_model


def test_linear_model():
    model = LinearModel(["y"], [1, 2, 3], 10)
    assert model.names == ["y"]
    eval(model.__repr__())

    # de-/serialize
    json.dumps(model.to_config())
    make_model(**model.to_config())

    # evaluate,
    X = pd.DataFrame(np.random.rand(10, 3), columns=["x1", "x2", "x3"])
    y = model.eval(X)
    assert y.columns == ["y"]
    assert len(y) == 10


def test_models():
    model1 = LinearModel(["cost"], [1, 2, 3])
    model2 = LinearModel(["pcf"], [5, 2.1, 0.9])
    models = Models([model1, model2])

    assert models[0] == model1
    assert models[1] == model2
    assert len(models) == 2
    assert models.names == ["cost", "pcf"]

    json.dumps(models.to_config())
    models = Models(models.to_config())

    # evaluate
    X = pd.DataFrame(np.random.rand(10, 3), columns=["x1", "x2", "x3"])
    Y = models.eval(X)
    assert list(Y.columns) == ["cost", "pcf"]
    assert len(Y) == 10


def test_custom_model():
    D = 5

    # custom sklearn model
    X_train, Y_train = make_regression(
        n_features=D, n_informative=4, n_targets=3, n_samples=20
    )
    pls = PLSRegression(n_components=4)
    pls.fit(X_train, Y_train)
    model1 = Model(names=["y1", "y2", "y3"])
    model1.eval = lambda x: pd.DataFrame(
        pls.predict(x.values), columns=model1.names, index=x.index
    )

    # linear model
    model2 = LinearModel(["cost"], [1, 2, 3, 4, 5])

    models = Models([model1, model2])
    assert models.names == ["y1", "y2", "y3", "cost"]

    # evaluate
    X = pd.DataFrame(np.random.rand(10, D), columns=[f"x{i}" for i in range(D)])
    Y = models.eval(X)
    assert list(Y.columns) == ["y1", "y2", "y3", "cost"]
    assert len(Y) == 10

    # to config, the custom model is omitted
    assert len(models.to_config()) == 1
