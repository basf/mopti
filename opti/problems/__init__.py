# flake8: noqa
from opti.problems.datasets import (
    HPLC,
    Alkox,
    BaumgartnerAniline,
    BaumgartnerBenzamide,
    Benzylation,
    Cake,
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
    Detergent_TwoOutputConstraints,
)
from opti.problems.mixed import DiscreteFuelInjector, DiscreteVLMOP2
from opti.problems.multi import (
    Daechert1,
    Daechert2,
    Daechert3,
    Hyperellipsoid,
    OmniTest,
    Poloni,
    Qapi1,
    WeldedBeam,
)
from opti.problems.single import (
    Ackley,
    Branin,
    Himmelblau,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    ThreeHumpCamel,
    Zakharov,
    Zakharov_Categorical,
    Zakharov_Constrained,
    Zakharov_NChooseKConstraint,
)
from opti.problems.univariate import Line1D, Parabola1D, Sigmoid1D, Sinus1D, Step1D
from opti.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
