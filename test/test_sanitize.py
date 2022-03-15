import numpy as np
import pandas as pd
import pytest

from opti.constraint import LinearEquality
from opti.objective import CloseToTarget, Maximize, Minimize
from opti.parameter import Continuous
from opti.problem import Problem
from opti.problems import Cake, Photodegradation, Zakharov_Constrained
from opti.tools import sanitize_problem


@pytest.mark.filterwarnings("ignore:f is not sanitized but dropped")
def test_sanitize_problem():
    def _test(original):
        sanitized = sanitize_problem(original)

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

    _test(Cake())

    _test(Photodegradation())

    z = Zakharov_Constrained()
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
                "mega_goal": np.sum(df**2, axis=1),
                "super_goal": np.linalg.norm(df, axis=1) ** 0.5,
            }
        ),
    )
    tp.create_initial_data(n_samples=100)
    _test(tp)

    empty = Problem([], [])  # this should not be possible
    with pytest.raises(TypeError):
        sanitize_problem(empty)
