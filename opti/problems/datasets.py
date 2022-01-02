"""
Chemical datasets.
These problems contain observed data but don't come with a ground truth.
"""
import os

import pandas as pd

from opti.constraint import LinearEquality, NChooseK
from opti.objective import CloseToTarget, Maximize, Minimize
from opti.parameter import Categorical, Continuous
from opti.problem import Problem

cdir = os.path.dirname(os.path.realpath(__file__))


def get_data(fname: str):
    return pd.read_csv(f"{cdir}/data/{fname}")


class Cake(Problem):
    """Cake recipe optimization with mixed objectives."""

    def __init__(self):
        super().__init__(
            name="Cake",
            inputs=[
                Continuous("wheat_flour", domain=[0, 1]),
                Continuous("spelt_flour", domain=[0, 1]),
                Continuous("sugar", domain=[0, 1]),
                Continuous("chocolate", domain=[0, 1]),
                Continuous("nuts", domain=[0, 1]),
                Continuous("carrot", domain=[0, 1]),
            ],
            outputs=[
                Continuous("calories", domain=[300, 600]),
                Continuous("taste", domain=[0, 5]),
                Continuous("browning", domain=[0, 2]),
            ],
            objectives=[
                Minimize("calories"),
                Maximize("taste"),
                CloseToTarget("browning", target=1.4),
            ],
            constraints=[
                LinearEquality(
                    [
                        "wheat_flour",
                        "spelt_flour",
                        "sugar",
                        "chocolate",
                        "nuts",
                        "carrot",
                    ],
                    rhs=1,
                )
            ],
            data=get_data("cake.csv"),
        )


class Alkox(Problem):
    """Alkoxylation dataset

    This dataset reports the biocatalytic oxidation of benzyl alcohol by a copper radical oxidase (AlkOx).
    The effects of enzyme loading, cocatalyst loading, and pH balance on both initial rate and total conversion were assayed.
    The dataset includes 104 samples with four parameters and one objective.

    Reference:
        F. Häse, M. Aldeghi, R.J. Hickman, L.M. Roch, M. Christensen, E. Liles, J.E. Hein, A. Aspuru-Guzik. Olympus: a benchmarking framework for noisy optimization and experiment planning. arXiv (2020), 2010.04153.
        [DOI](https://doi.org/10.1088/2632-2153/abedc8).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="Alkox",
            inputs=[
                Continuous("residence_time", domain=[0.05, 1]),
                Continuous("ratio", domain=[0.5, 10]),
                Continuous("concentration", domain=[2, 8]),
                Continuous("temperature", domain=[6, 8]),
            ],
            outputs=[Continuous("conversion")],
            objectives=[Maximize("conversion")],
            data=get_data("alkox.csv"),
        )


class BaumgartnerAniline(Problem):
    """Aniline C-N cross-coupling dataset.

    Reference:
        Baumgartner et al. 2019 - Use of a Droplet Platform To Optimize Pd-Catalyzed C-N Coupling Reactions Promoted by Organic Bases
        [DOI](https://doi.org/10.1021/acs.oprd.9b00236).
        Data obtained from [Summit](https://github.com/sustainable-processes/summit).
    """

    def __init__(self):
        super().__init__(
            name="Aniline cross-coupling, Baumgartner 2019",
            inputs=[
                Categorical("catalyst", domain=["tBuXPhos", "tBuBrettPhos", "AlPhos"]),
                Categorical("base", domain=["TEA", "TMG", "BTMG", "DBU"]),
                Continuous("base_equivalents", domain=[1.0, 2.5]),
                Continuous("temperature", domain=[30, 100]),
                Continuous("residence_time", domain=[60, 1800]),
            ],
            outputs=[Continuous("yield", domain=[0, 1])],
            objectives=[Maximize("yield")],
            data=get_data("baumgartner_aniline.csv"),
        )


class BaumgartnerBenzamide(Problem):
    """Benzamide C-N cross-coupling dataset.

    Reference:
        Baumgartner et al. 2019 - Use of a Droplet Platform To Optimize Pd-Catalyzed C-N Coupling Reactions Promoted by Organic Bases
        [DOI](https://doi.org/10.1021/acs.oprd.9b00236).
        Data obtained from [Summit](https://github.com/sustainable-processes/summit).
    """

    def __init__(self):
        super().__init__(
            name="Benzamide cross-coupling, Baumgartner 2019",
            inputs=[
                Categorical("catalyst", domain=["tBuXPhos", "tBuBrettPhos"]),
                Categorical("base", domain=["TMG", "BTMG", "DBU", "MTBD"]),
                Continuous("base_equivalents", domain=[1.0, 2.1]),
                Continuous("temperature", domain=[30, 100]),
                Continuous("residence_time", domain=[60, 1850]),
            ],
            outputs=[Continuous("yield", domain=[0, 1])],
            objectives=[Maximize("yield")],
            data=get_data("baumgartner_benzamide.csv"),
        )


class Benzylation(Problem):
    """Benzylation dataset.

    This dataset reports the yield of undesired product (impurity) in an N-benzylation reaction.
    Four conditions of this reaction performed in a flow reactor can be controlled to minimize the yield of impurity.
    The dataset includes 73 samples with four parameters and one objective.

    Reference:
        A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.
        [DOI](https://doi.org/10.1016/j.cej.2018.07.031).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="Benzylation",
            inputs=[
                Continuous("flow_rate", domain=[0.2, 0.4]),
                Continuous("ratio", domain=[1.0, 5.0]),
                Continuous("solvent", domain=[0.5, 1.0]),
                Continuous("temperature", domain=[110.0, 150.0]),
            ],
            outputs=[Continuous("impurity")],
            objectives=[Minimize("impurity")],
            data=get_data("benzylation.csv"),
        )


class Fullerenes(Problem):
    """Buckminsterfullerene dataset.

    This dataset reports the production of o-xylenyl adducts of Buckminsterfullerenes.
    Three process conditions (temperature, reaction time and ratio of sultine to C60) are varied to maximize the mole fraction of the desired product.
    Experiments are executed on a three factor fully factorial grid with six levels per factor.
    The dataset includes 246 samples with three parameters and one objective.

    Reference:
        B.E. Walker, J.H. Bannock, A.M. Nightingale, J.C. deMello. Tuning reaction products by constrained optimisation. React. Chem. Eng., (2017), 2, 785-798.
        [DOI](https://doi.org/10.1039/C7RE00123A).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="Fullerenes",
            inputs=[
                Continuous("reaction_time", domain=[3.0, 31.0]),
                Continuous("sultine", domain=[1.5, 6.0]),
                Continuous("temperature", domain=[100.0, 150.0]),
            ],
            outputs=[Continuous("product")],
            objectives=[Maximize("product")],
            data=get_data("fullerenes.csv"),
        )


class HPLC(Problem):
    """High-performance liquid chromatography dataset.

    This dataset reports the peak response of an automated high-performance liquid chromatography (HPLC) system for varying process parameters.
    The dataset includes 1,386 samples with six parameters and one objective.

    Reference:
        L.M. Roch, F. Häse, C. Kreisbeck, T. Tamayo-Mendoza, L.P.E. Yunker, J.E. Hein, A. Aspuru-Guzik. ChemOS: an orchestration software to democratize autonomous discovery. (2018)
        [DOI](https://doi.org/10.26434/chemrxiv.5953606.v1).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="HPLC",
            inputs=[
                Continuous("sample_loop", domain=[0.0, 0.08]),
                Continuous("additional_volume", domain=[0.0, 0.06]),
                Continuous("tubing_volume", domain=[0.1, 0.9]),
                Continuous("sample_flow", domain=[0.5, 2.5]),
                Continuous("push_speed", domain=[80.0, 150]),
                Continuous("wait_time", domain=[0.5, 10.0]),
            ],
            outputs=[Continuous("peak_area")],
            objectives=[Maximize("peak_area")],
            data=get_data("hplc.csv"),
        )


class Photodegradation(Problem):
    """Photodegration dataset.

    This dataset reports the degradation of polymer blends for organic solar cells under the exposure to light.
    Individual data points encode the ratios of individual polymers in one blend, along with the measured photodegradation of this blend.
    The dataset includes 2,080 samples with five parameters and one objective.

    Reference:
        S. Langner, F. Häse, J.D. Perea, T. Stubhan, J. Hauch, L.M. Roch, T. Heumueller, A. Aspuru-Guzik, C.J. Brabec. Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multicomponent Systems. Advanced Materials, 2020, 1907801.
        [DOI](https://doi.org/10.1002/adma.201907801).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="Photodegradation",
            inputs=[
                Continuous("PCE10", domain=[0, 1]),
                Continuous("WF3", domain=[0, 1]),
                Continuous("P3HT", domain=[0, 1]),
                Continuous("PCBM", domain=[0, 1]),
                Continuous("oIDTBR", domain=[0, 1]),
            ],
            outputs=[Continuous("degradation")],
            objectives=[Minimize("degradation")],
            constraints=[
                LinearEquality(
                    ["PCE10", "WF3", "P3HT", "PCBM", "oIDTBR"], rhs=1, lhs=1
                ),
                NChooseK(["PCE10", "WF3"], max_active=1),
            ],
            data=get_data("photodegradation.csv"),
        )


class ReizmanSuzuki(Problem):
    """Suzuki-Miyaura cross-coupling optimization.

    Each case was has a different set of substrates but the same possible catalysts.

    Reference:
        Reizman et al. (2016) Suzuki-Miyaura cross-coupling optimization enabled by automated feedback. Reaction chemistry & engineering, 1(6), 658-666
        [DOI](https://doi.org/10.1039/C6RE00153J).
        Data obtained from [Summit](https://github.com/sustainable-processes/summit).
    """

    def __init__(self, case=1):
        assert case in [1, 2, 3, 4]
        super().__init__(
            name=f"Reizman 2016 - Suzuki Case {case}",
            inputs=[
                Categorical(
                    "catalyst",
                    domain=[
                        "P1-L1",
                        "P2-L1",
                        "P1-L2",
                        "P1-L3",
                        "P1-L4",
                        "P1-L5",
                        "P1-L6",
                        "P1-L7",
                    ],
                ),
                Continuous("t_res", domain=[60, 600]),
                Continuous("temperature", domain=[30, 110]),
                Continuous("catalyst_loading", domain=[0.496, 2.515]),
            ],
            outputs=[
                Continuous("ton", domain=[0, 100]),
                Continuous("yield", domain=[0, 100]),
            ],
            objectives=[Maximize("ton"), Maximize("yield")],
            data=get_data(f"reizman_suzuki{case}.csv"),
        )


class SnAr(Problem):
    """SnAr reaction optimization.

    This dataset reports the e-factor for a nucleophilic aromatic substitution following the SnAr mechanism.
    Individual data points encode four process parameters for a flow reactor to run the reaction, along with the measured e-factor (defined as the ratio of the mass waste to the mass of product).
    The dataset includes 67 samples with four parameters and one objective.

    Reference:
        A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.
        [DOI](https://doi.org/10.1016/j.cej.2018.07.031).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="SnAr",
            inputs=[
                Continuous("residence_time", domain=[0.5, 2.0]),
                Continuous("ratio", domain=[1.0, 5.0]),
                Continuous("concentration", domain=[0.1, 0.5]),
                Continuous("temperature", domain=[60.0, 140.0]),
            ],
            outputs=[Continuous("impurity")],
            objectives=[Minimize("impurity")],
            data=get_data("snar.csv"),
        )


class Suzuki(Problem):
    """Suzuki reaction dataset.

    This dataset reports palladium-catalyzed Suzuki cross-coupling between 2-bromophenyltetrazole and an electron-deficient aryl boronate.
    Four reaction conditions can be controlled to maximise the reaction yield.
    The dataset includes 247 samples with four parameters and one objective.

    Reference:
        F. Häse, M. Aldeghi, R.J. Hickman, L.M. Roch, M. Christensen, E. Liles, J.E. Hein, A. Aspuru-Guzik. Olympus: a benchmarking framework for noisy optimization and experiment planning. arXiv (2020), 2010.04153.
        [DOI](https://doi.org/10.1088/2632-2153/abedc8).
        Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
    """

    def __init__(self):
        super().__init__(
            name="Suzuki",
            inputs=[
                Continuous("temperature", domain=[75.0, 90.0]),
                Continuous("pd_mol", domain=[0.5, 5.0]),
                Continuous("arbpin", domain=[1.0, 1.8]),
                Continuous("k3po4", domain=[1.5, 3.0]),
            ],
            outputs=[Continuous("yield")],
            objectives=[Maximize("yield")],
            data=get_data("suzuki.csv"),
        )
