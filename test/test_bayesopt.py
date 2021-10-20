from opti.bayesopt.algorithm import Algorithm
from opti.problems import ZDT1


def test_init():
    # set up with initial data from problem
    problem = ZDT1(n_inputs=3)
    problem.create_initial_data(6)
    optimizer = Algorithm(problem)
    assert len(optimizer.problem.data) == 6
