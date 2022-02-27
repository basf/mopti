from opti.constraint import LinearInequality
from opti.problems import Cake
from opti.tools import subspace_problem


def test_cake():
    problem = Cake()
    sub = subspace_problem(problem)

    # 6 parameters are transformed to 5
    assert len(sub.inputs) == 5

    # 1 linear equality is transformed to 12 linear inequalities
    assert len(sub.constraints) == 12
    for c in sub.constraints:
        assert isinstance(c, LinearInequality)

    # # the spaces are equivalent
    # Xt = sub.sample_inputs(1000)
    # sub.inverse_transform(Xt)
