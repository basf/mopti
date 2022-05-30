import numpy as np
import pandas as pd

from opti.constraint import LinearInequality, NonlinearInequality
from opti.objective import Minimize
from opti.parameter import Continuous
from opti.problem import Problem

class Gardner(Problem):
# 2D Gardner problem with 1 constraint
# From Gardner et al. 2014. http://proceedings.mlr.press/v32/gardner14.pdf
    def __init__(self):
        super().__init__(
            name="Gardner(d=2,p=1)",
            inputs=[
                Continuous("x1", domain=[0, 2.0 * np.pi]),
                Continuous("x2", domain=[0, 2.0 * np.pi]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[NonlinearInequality("sin(x1) * sin(x2) + 0.95")],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("sin(x1) + x2"),},
            index=X.index,
        )
    
    def get_optima(self) -> pd.DataFrame:
        x = np.array([[1.5 * np.pi, np.arcsin(0.95)]])
        y = np.array([[np.arcsin(0.95) - 1.0]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Gramacy(Problem):
# 2D Gramacy problem with 2 constraints
# From Gramacy et al. 2016. https://arxiv.org/pdf/1403.4890.pdf
    def __init__(self):
        super().__init__(
            name="Gramacy(d=2,p=2)",
            inputs=[
                Continuous("x1", domain=[0, 1.0]),
                Continuous("x2", domain=[0, 1.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("1.5 - x1 - 2.0 * x2 - 0.5 * sin(2.0 * arccos(-1.0) * (x1**2 - 2.0 * x2))"),
                NonlinearInequality("x1**2 + x2**2 - 1.5")
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("x1 + x2"),},
            index=X.index,
        )
    
    def get_optima(self) -> pd.DataFrame:
        x = np.array([[0.1954, 0.4044]])
        y = np.array([[0.5998]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Sasena(Problem):
# 2D Sasena problem with 3 constraints
# From Sasena's PhD thesis 2002. https://www.mat.univie.ac.at/~neum/glopt/mss/Sas02.pdf 
    def __init__(self):
        super().__init__(
            name="Sasena(d=2,p=3)",
            inputs=[
                Continuous("x1", domain=[0, 1.0]),
                Continuous("x2", domain=[0, 1.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("((x1 - 3.0)**2 + (x2 + 2.0)**2) * exp(- x2**7) - 12.0"),
                LinearInequality(["x1", "x2"], lhs=[10.0, 1.0], rhs=7.0),
                NonlinearInequality("(x1 - 0.5)**2 + (x2 - 0.5)**2 - 0.2")
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("-(x1 - 1.0)**2 - (x2 - 0.5)**2"),},
            index=X.index,
        )
    
    def get_optima(self) -> pd.DataFrame:
        x = np.array([[0.2017, 0.8332]])
        y = np.array([[-0.7483]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class G4(Problem):
# 5D G4 problem with 6 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G4(d=5,p=6)",
            inputs=[
                Continuous("x1", domain=[78.0, 102.0]),
                Continuous("x2", domain=[33.0, 45.0]),
                Continuous("x3", domain=[27.0, 45.0]),
                Continuous("x4", domain=[27.0, 45.0]),
                Continuous("x5", domain=[27.0, 45.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("(85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5) - 92.0"),
                NonlinearInequality("- (85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5)"),
                NonlinearInequality("(80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2) - 110.0"),
                NonlinearInequality("90.0 - (80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2)"),
                NonlinearInequality("(9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4) - 25.0"),
                NonlinearInequality("20.0 - (9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4)")
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141"),},
            index=X.index,
        )
    
    def get_optima(self) -> pd.DataFrame:
        x = np.array([[78., 33., 29.995256025682, 45., 36.775812905788]])
        y = np.array([[-30665.539]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class G6(Problem):
# 2D G6 problem with 2 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G6(d=2,p=2)",
            inputs=[
                Continuous("x1", domain=[13.5, 14.5]),
                Continuous("x2", domain=[0.5, 1.5]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("-(x1 - 5.0)**2 - (x2 - 5.0)**2 + 100.0"),
                NonlinearInequality("(x1 - 6.0)**2 + (x2 - 5.0)**2 - 82.81"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("(x1 - 10.0)**3 + (x2 - 20.0)**3"),},
            index=X.index,
        )
    
    def get_optima(self) -> pd.DataFrame:
        x = np.array([[14.095, 0.84296]])
        y = np.array([[-6961.81388]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class G7(Problem):
# 10D G7 problem with 8 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G7(d=10,p=8)",
            inputs=[
                Continuous(f"x{i+1}", domain=[-10.0, 10.0]) for i in range(10)
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                LinearInequality(["x1", "x2", "x7", "x8"], lhs=[4.0, 5.0, -3.0, 9.0], rhs=105.0),
                LinearInequality(["x1", "x2", "x7", "x8"], lhs=[10.0, -8.0, -17.0, 2.0], rhs=0),
                LinearInequality(["x1", "x2", "x9", "x10"], lhs=[-8.0, 2.0, 5.0, -2.0], rhs=12.0),
                NonlinearInequality("3.0 * (x1 - 2.0)**2 + 4.0 * (x2 - 3.0)**2 + 2.0 * x3**2 - 7.0 * x4 - 120.0"),
                NonlinearInequality("5.0 * x1**2 + 8.0 * x2 + (x3 - 6.0)**2 - 2.0 * x4 - 40.0"),
                NonlinearInequality("0.5 * (x1 - 8.0)**2 + 2.0 * (x2 - 4.0)**2 + 3.0 * x5**2 - x6 - 30.0"),
                NonlinearInequality("x1**2 + 2.0 * (x2 - 2.0)**2 - 2.0 * x1 * x2 + 14.0 * x5 - 6.0 * x6"),
                NonlinearInequality("- 3.0 * x1 + 6.0 * x2 + 12.0 * (x9 - 8.0)**2 - 7.0 * x10"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("x1**2 + x2**2 + x1 * x2 - 14.0 * x1 - 16.0 * x2 + (x3 - 10.0)**2 + 4.0 * (x4 - 5.0)**2 + (x5 - 3.0)**2 + 2.0 * (x6 - 1.0)**2 + 5.0 * x7**2 + 7.0 * (x8 - 11.0)**2 + 2.0 * (x9 - 10.0)**2 + (x10 - 7.0)**2 + 45.0"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[2.171996, 2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927]])
        y = np.array([[24.3062091]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)

class G8(Problem):
# 2D G8 problem with 2 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G8(d=2,p=2)",
            inputs=[
                Continuous("x1", domain=[0.5, 10.0]),
                Continuous("x2", domain=[0.5, 10.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("x1**2 - x2 + 1.0"),
                NonlinearInequality("1.0 - x1 + (x2 - 4.0)**2"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("- (sin(2.0 * arccos(-1.0) * x1)**3 * sin(2.0 * arccos(-1.0) * x2)) / (x1**3 * (x1 + x2))"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[1.2279713, 4.2453733]])
        y = np.array([[-0.095825]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class G9(Problem):
# 7D G9 problem with 4 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G9(d=7,p=4)",
            inputs=[
                Continuous(f"x{i+1}", domain=[-10.0, 10.0]) for i in range(7)
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("2.0 * x1**2 + 3.0 * x2**4 + x3 + 4.0 * x4**2 + 5.0 * x5 - 127.0"),
                NonlinearInequality("7.0 * x1 + 3.0 * x2 + 10.0 * x3**2 + x4 - x5 - 282.0"),
                NonlinearInequality("23.0 * x1 + x2**2 + 6.0 * x6**2 - 8.0 * x7 - 196.0"),
                NonlinearInequality("4.0 * x1**2 + x2**2 - 3.0 * x1 * x2 + 2.0 * x3**2 + 5.0 * x6 - 11.0 * x7"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("(x1 - 10.0)**2 + 5.0 * (x2 - 12.0)**2 + x3**4 + 3.0 * (x4 - 11.0)**2 + 10.0 * x5**6 + 7.0 * x6**2 + x7**4 - 4.0 * x6 * x7 - 10.0 * x6 - 8.0 * x7"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]])
        y = np.array([[680.6300573]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class G10(Problem):
# 8D G10 problem with 6 constraints
# From Michalewicz and Schoenauer 1996. https://ieeexplore.ieee.org/document/6791784
    def __init__(self):
        super().__init__(
            name="G10(d=8,p=6)",
            inputs=[
                Continuous("x1", domain=[100.0, 10000.0]),
                Continuous("x2", domain=[1000.0, 10000.0]),
                Continuous("x3", domain=[1000.0, 10000.0]),
                Continuous("x4", domain=[10.0, 1000.0]),
                Continuous("x5", domain=[10.0, 1000.0]),
                Continuous("x6", domain=[10.0, 1000.0]),
                Continuous("x7", domain=[10.0, 1000.0]),
                Continuous("x8", domain=[10.0, 1000.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                LinearInequality(["x4", "x6"], lhs=[0.0025, 0.0025], rhs=1.0),
                LinearInequality(["x4", "x5", "x7"], lhs=[-0.0025, 0.0025, 0.0025], rhs=1.0),
                LinearInequality(["x5", "x8"], lhs=[-0.01, 0.01], rhs=1.0),
                NonlinearInequality("100.0 * x1 - x1 * x6 + 833.33252 * x4 - 83333.333"),
                NonlinearInequality("x2 * x4 - x2 * x7 - 1250.0 * x4 + 1250.0 * x5"),
                NonlinearInequality("x3 * x5 - x3 * x8 - 2500.0 * x5 + 1250000.0"),
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("x1 + x2 + x3"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799, 286.4162, 395.5979]])
        y = np.array([[7049.3307]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Tension_Compression(Problem):
# The 3D tension-compression string design problem aims to minimize the weight of a tension/compression spring under 4 mechanical constraints
# From Coello and Mezura-Montes 2002. https://link.springer.com/chapter/10.1007/978-0-85729-345-9_23 
    def __init__(self):
        super().__init__(
            name="Tension_Compression(d=3,p=4)",
            inputs=[
                Continuous("x1", domain=[2.0, 15.0]),
                Continuous("x2", domain=[0.25, 1.3]),
                Continuous("x3", domain=[0.05, 2.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("1.0 - x2**3 * x1 / (71785.0 * x3**4)"),
                NonlinearInequality("(4.0 * x2**2 - x2 * x3) / (12566.0 * x3**3 * (x2 - x3)) + 1.0 / (5108.0 * x3**2) - 1.0"),
                NonlinearInequality("1.0 - 140.45 * x3 / (x1 * x2**2)"),
                NonlinearInequality("(x2 + x3) / 1.5 - 1.0"),
            ],
        )
                
    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("x3**2 * x2 * (x1 + 2.0)"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[11.21390736278739, 0.35800478345599, 0.05174250340926]])
        y = np.array([[0.012666]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Pressure_Vessel(Problem):
# The 4D pressure vessel design problem  aims to minimize the cost of designing a cylindrical vessel subject to 4 constraints. 
# The original problem did not give the bounds for design variables. 
# Here we use the bounds in [1] that contains the best known solution found in [2]. 
# Note that the fourth constraint always holds after bounding the input variables. Therefore, we remove it. 
# An interesting setting is that the first two variables (x1 and x2) have to be multiples of 0.0625 and are rounded to the closest such value before evaluating the objective and constraints.
# However, the NonlinearInequality class in mopti doesn't support 'round'. Here I remove this setting (By Shiqiang)
# [1] Eriksson and Poloczek 2021. http://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf   
# [2] Coello and Mezura-Montes 2002. https://link.springer.com/chapter/10.1007/978-0-85729-345-9_23 
    def __init__(self):
        super().__init__(
            name="Pressure_Vessel(d=4,p=3)",
            inputs=[
                Continuous("x1", domain=[0, 10.0]),
                Continuous("x2", domain=[0, 10.0]),
                Continuous("x3", domain=[10.0, 50.0]),
                Continuous("x4", domain=[150.0, 200.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("- x1 + 0.0193 * x3"),
                NonlinearInequality("- x2+ 0.00954 * x3"),
                NonlinearInequality("- arccos(-1.0) * x3**2 * x4 - 4.0 * arccos(-1.0) / 3.0 * x3**3 + 1296000.0"),
            ],
        )
    
    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[0.8125, 0.4375, 42.0984, 176.6368]])
        y = np.array([[6059.715]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Welded_Beam(Problem):
# In 4D welded beam design problem, the cost of design a welded beam is minimized subject to 5 constraints. 
# The original version of this problem in [1] had 7 constraints, 2 of which could be shrunk into the bounds. 
# Here we use the formulations in [2].
# [1] Coello and Mezura-Montes 2002. https://link.springer.com/chapter/10.1007/978-0-85729-345-9_23 
# [2] Hedar and Fukushima 2006. https://link.springer.com/article/10.1007/s10898-005-3693-z
    def __init__(self):
        super().__init__(
            name="Welded_Beam(d=4,p=5)",
            inputs=[
                Continuous("x1", domain=[0.125, 10.0]),
                Continuous("x2", domain=[0.1, 10.0]),
                Continuous("x3", domain=[0.1, 10.0]),
                Continuous("x4", domain=[0.1, 10.0]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("((6000.0 / (2.0**0.5 * x1 * x2))**2 + ((6000. * (14.0 + 0.5 * x2) * (0.25 * (x2**2 + (x1 + x3)**2))**0.5) / (2.0 * (0.707 * x1 * x2 * (x2**2 / 12.0 + 0.25 * (x1 + x3)**2))))**2 + x2 * (6000.0 / (2.0**0.5 * x1 * x2)) * ((6000. * (14.0 + 0.5 * x2) * (0.25 * (x2**2 + (x1 + x3)**2))**0.5) / (2.0 * (0.707 * x1 * x2 * (x2**2 / 12.0 + 0.25 * (x1 + x3)**2)))) / (0.25 * (x2**2 + (x1 + x3)**2))**0.5) ** 0.5 - 13000.0"),
                NonlinearInequality("504000.0 / (x3**2 * x4) - 30000.0"),
                LinearInequality(["x1", "x4"], lhs=[1.0, -1.0], rhs=0),
                NonlinearInequality("6000.0 - 64746.022 * (1.0 - 0.0282346 * x3) * x3 * x4**3"),
                NonlinearInequality("2.1952 / (x3**3 * x4) - 0.25"),
            ],
        )
        
    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[0.24435257, 6.2157922, 8.2939046, 0.24435257]])
        y = np.array([[2.381065]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)
    
class Speed_Reducer(Problem):
# The goal of 7D speed reducer problem is to minimize the weight of a speed reducer under 11 mechanical constraints. 
# The third variable is a category variable. However, regarding it as a continuous variable does not change the optimum.
# From Coello and Mezura-Montes 2002. https://link.springer.com/chapter/10.1007/978-0-85729-345-9_23 
    def __init__(self):
        super().__init__(
            name="Speed_Reducer(d=7,p=11)",
            inputs=[
                Continuous("x1", domain=[2.6, 3.6]),
                Continuous("x2", domain=[0.7, 0.8]),
                Continuous("x3", domain=[17.0, 28.0]),
                Continuous("x4", domain=[7.3, 8.3]),
                Continuous("x5", domain=[7.8, 8.3]),
                Continuous("x6", domain=[2.9, 3.9]),
                Continuous("x7", domain=[4.9, 5.9]),
            ],
            outputs=[Continuous("y0")],
            objectives=[Minimize("y0")],
            constraints=[
                NonlinearInequality("27.0 / (x1 * x2**2 * x3) - 1.0"),
                NonlinearInequality("397.5 / (x1 * x2**2 * x3**2) - 1.0"),
                NonlinearInequality("1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0"),
                NonlinearInequality("1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0"),
                NonlinearInequality("((745.0 * x4 / (x2 * x3))**2 + 16900000.0)**0.5 / x6**3 - 110.0"),
                NonlinearInequality("((745.0 * x5 / (x2 * x3))**2 + 157500000.0)**0.5 / x7**3 - 85.0"),
                NonlinearInequality("x2 * x3 - 40.0"),
                NonlinearInequality("- x1 / x2 + 5.0"),
                NonlinearInequality("x1 / x2 - 12.0"),
                NonlinearInequality("(1.5 * x6 + 1.9) / x4 - 1.0"),
                NonlinearInequality("(1.1 * x7 + 1.9) / x5 - 1.0"),
                
            ],
        )

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"y0": X.eval("0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) - 1.508 * x1 * (x6**2 + x7**2) + 7.4777 * (x6**3 + x7**3) + 0.7854 * (x4 * x6**2 + x5 * x7**2)"),},
            index=X.index,
        )

    def get_optima(self) -> pd.DataFrame:
        x = np.array([[3.5, 0.7, 17., 7.3, 7.8, 3.350215, 5.286683]])
        y = np.array([[2996.3482]])
        return pd.DataFrame(np.c_[x, y], columns=self.inputs.names + self.outputs.names)