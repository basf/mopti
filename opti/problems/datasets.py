import os

import pandas as pd

from opti.constraint import LinearEquality, NChooseK
from opti.objective import Maximize, Minimize
from opti.parameter import Categorical, Continuous
from opti.problem import Problem

cdir = os.path.dirname(os.path.realpath(__file__))


def get_data(fname: str):
    return pd.read_csv(f"{cdir}/data/{fname}")


class Alkox(Problem):
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
    def __init__(self):
        super().__init__(
            name="Baumgartner 2019 - Aniline Cross-Coupling",
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
    def __init__(self):
        super().__init__(
            name="Baumgartner 2019 - Benzamide Cross-Coupling",
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
