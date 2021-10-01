# flake8: noqa
from opti.problems.baking import Bread, Cake
from opti.problems.benchmark import (
    Daechert1,
    Daechert2,
    Daechert3,
    Hyperellipsoid,
    Qapi1,
)
from opti.problems.datasets import (
    HPLC,
    Alkox,
    BaumgartnerAniline,
    BaumgartnerBenzamide,
    Benzylation,
    Fullerenes,
    Photodegradation,
    ReizmanSuzuki,
    SnAr,
    Suzuki,
)
from opti.problems.detergent import (
    Detergent,
    Detergent_NChooseKConstraint,
    Detergent_OutputConstraint,
)
from opti.problems.mixed import DiscreteFuelInjector, DiscreteVLMOP2
from opti.problems.noisify import (
    noisify_problem,
    noisify_problem_with_gaussian,
    noisify_problem_with_scipy_stats,
)
from opti.problems.single import (
    Ackley,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Zakharov,
    Zakharov_Categorical,
    Zakharov_Constrained,
    Zakharov_NChooseKConstraint,
)
from opti.problems.univariate import Line1D, Parabola1D, Sigmoid1D, Sinus1D, Step1D
from opti.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
