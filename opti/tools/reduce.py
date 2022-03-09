from logging.handlers import WatchedFileHandler
import numpy as np
import pandas as pd                                                     
from typing import Callable, Dict, List, Optional, Tuple, Union


from opti import Problem
from opti.constraint import Constraints, LinearEquality, LinearInequality
from opti.objective import Objectives
from opti.parameter import Categorical, Parameters
from sympy import Matrix


#TODO: Wie geht man mit diskreten inputs um? --> einfach domain beibehalten?
#TODO: f für das reduzierte problem anpassen
#TODO: Was bedeutet das attribut Models in class Problem?
#TODO: tests schreiben

################################
import json
import os
import re

from opti.constraint import Constraint, Constraints
from opti.model import Model, Models
from opti.objective import Objective, Objectives
from opti.parameter import Categorical, Parameter, Parameters

ParametersLike = Union[Parameters, List[Parameter], List[Dict]]
ObjectivesLike = Union[Objectives, List[Objective], List[Dict]]
ConstraintsLike = Union[Constraints, List[Constraint], List[Dict]]
ModelsLike = Union[Models, List[Model], List[Dict]]
DataFrameLike = Union[pd.DataFrame, Dict]
PathLike = Union[str, bytes, os.PathLike]

class ReducedProblem(Problem):
    def __init__(
        self,
        inputs: ParametersLike,
        outputs: ParametersLike,
        objectives: Optional[ObjectivesLike] = None,
        constraints: Optional[ConstraintsLike] = None,
        output_constraints: Optional[ObjectivesLike] = None,
        f: Optional[Callable] = None,
        models: Optional[ModelsLike] = None,
        data: Optional[DataFrameLike] = None,
        optima: Optional[DataFrameLike] = None,
        name: Optional[str] = None,
        equalities: Optional[List[str]] = None,
        **kwargs,
    ):
        """An optimization problem.

        Args:
            inputs: Input parameters.
            outputs: Output parameters.
            objectives: Optimization objectives. Defaults to minimization.
            constraints: Constraints on the inputs.
            output_constraints: Constraints on the outputs.
            f: Function to evaluate the outputs for given inputs.
                Must have the signature: f(x: pd.DataFrame) -> pd.DataFrame
            data: Experimental data.
            optima: Pareto optima.
            name: Name of the problem.
            equalities: Only in case of problem reduction due to equality con
                straints. Used to augment the solution of the reduced problem.
        """
        super().__init__(
            inputs, 
            outputs, 
            objectives, 
            constraints, 
            output_constraints, 
            f, 
            models, 
            data, 
            optima, 
            name
            )

        if isinstance(equalities, List):
            self.equalities = equalities
        elif equalities is not None:
            self.equalities = eval(equalities)
        else:
            self.equalities = None 

        #check if the names in self.equalities are valid
        if self.n_equalities > 0:
            for lhs, rhs in self.equalities:
                rhs = re.findall("(?<=\*\s)[^\*]*(?=\s\+)", rhs)
                for name in rhs:
                    if name not in self.inputs.names:
                        raise ValueError(f"Equality refers to unknown parameter: {name}")

    @property
    def n_equalities(self) -> int:
        return 0 if self.equalities is None else len(self.equalities)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = super().__str__()
        if self.equalities is not None:
            s = s[:-1] + f"equalities=\n{self.equalities}\n"
        return "Reduced" + s + ")"

    @staticmethod
    def from_config(config: dict) -> "ReducedProblem":
        """Create a Problem instance from a configuration dict."""
        return ReducedProblem(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        if self.equalities is not None:
            config["equalities"] = self.equalities
        return config

    @staticmethod
    def from_json(fname: PathLike) -> "ReducedProblem":
        """Read a problem from a JSON file."""
        with open(fname, "rb") as infile:
            config = json.loads(infile.read())
        return ReducedProblem(**config)

    def to_json(self, fname: PathLike) -> None:
        """Save a problem from a JSON file."""
        with open(fname, "wb") as outfile:
            b = json.dumps(self.to_config(), ensure_ascii=False, separators=(",", ":"))
            outfile.write(b.encode("utf-8"))


    
        

#####################################

def reduce(problem: Problem) -> ReducedProblem:
    """Reduce a problem with linear equality constraints and linear inequality constraints
    to a subproblem with linear inequality constraints and no linear equality constraints.

    Args:
        problem (Problem): problem to be reduced

    Returns:
        Reduced problem where linear equality constraints have been eliminated 
    
    """
    #check if the reduction can be applied or if a trivial case is present
    #Are there any constraints?
    if problem.constraints == None:
        return problem
    
    #find linear equality constraints and write them into a matrix/DataFrame A
    linearEqualityConstraints = []
    otherConstraints = []
    for c in problem.constraints:
        if isinstance(c, LinearEquality):
            linearEqualityConstraints.append(c)
        else:
            otherConstraints.append(c)
    linearEqualityConstraints = Constraints(linearEqualityConstraints)

    if len(linearEqualityConstraints) == 0:
        return problem
    
    #only consider non-categorical inputs
    inputs = []
    categoricalInputs = []
    for p in problem.inputs:
        if not isinstance(p, Categorical):
            inputs.append(p)
        else:
            categoricalInputs.append(p)
    inputs = Parameters(inputs)
    
    if len(inputs) == 0:
        return problem


    N = len(linearEqualityConstraints)
    M = len(inputs) + 1
    names = np.concatenate((inputs.names, ["rhs"]))
    
    A_aug = pd.DataFrame(data=np.zeros(shape=(N,M)), columns=names)
    
    for i in range(len(linearEqualityConstraints)):
        c = linearEqualityConstraints[i]

        A_aug.loc[i, c.names] = c.lhs
        A_aug.loc[i, "rhs"] = c.rhs
    A_aug = A_aug.values
    A = A_aug[:,:-1]
    b = A_aug[:,-1]

    #catch special cases
    rk_A_aug = np.linalg.matrix_rank(A_aug)
    rk_A = np.linalg.matrix_rank(A)

    if (rk_A == rk_A_aug):
        pass
    elif (rk_A < rk_A_aug):
        raise Warning("There is no solution fulfilling the linear equality constraints")
    else:
        raise Warning("Something went wrong. Rank of coefficient matrix must not be "
        "larger than rank of augmented coefficient matrix")

    if (rk_A == M-1):
        x = np.linalg.solve(A,b)
        raise Warning("There is a unique solution x for the linear inequalities: x=" + str(x))
            
    #bring A_aug to reduced row-echelon form
    A_aug_rref, pivots = Matrix(A_aug).rref()
    pivots = np.array(pivots)
    A_aug_rref = np.array(A_aug_rref).astype(np.float64)

    #formulate box bounds as linear inequality constraints in matrix form
    B = np.zeros(shape=(2*(M-1), M))
    B[:M-1, :M-1] = np.eye(M-1)
    B[M-1:, :M-1] = -np.eye(M-1)

    B[:M-1,-1] = inputs.bounds.loc["max"]
    B[M-1:,-1] = -inputs.bounds.loc["min"]

    #eliminate columns with pivot element
    for i in range(len(pivots)):
        p = pivots[i]
        B[p,:] -= A_aug_rref[i,:]
        B[p+M-1,:] += A_aug_rref[i,:]


    #build up reduced problem
    _inputs = categoricalInputs
    for i in range(len(inputs)):
        #add all inputs that were not eliminated
        if i not in pivots:
            _inputs.append(inputs[names[i]])
    _inputs = Parameters(_inputs)

    _constraints = otherConstraints
    for i in pivots:
        ind = np.where(B[i,:-1] != 0)[0]
        if len(ind)>0:
            c = LinearInequality(names=list(names[ind]),lhs=B[i,ind],rhs=B[i,-1])
            _constraints.append(c)
        else:
            if B[i,-1] < -1e-16:
                raise Warning("There is no solution in the domain of the variables "
                "that fulfills the constraints.")
        
        ind = np.where(B[i+M-1,:-1] != 0)[0]
        if len(ind)>0:
            c = LinearInequality(names=list(names[ind]),lhs=B[i+M-1,ind],rhs=B[i+M-1,-1])
            _constraints.append(c)
        else:
            if B[i+M-1,-1] < -1e-16:
                raise Warning("There is no solution in the domain of the variables "
                "that fulfills the constraints.")
    _constraints = Constraints(_constraints)

    equalities = []
    for i in range(len(pivots)):
        name = names[pivots[i]]
        lhs = ""
        
        for j in range(len(names)-1):
            if (A_aug_rref[i,j] != 0 and j != pivots[i]):
                lhs += "("+str(-A_aug_rref[i,j]) + ") * " + names[j] + " + "
            
        if A_aug_rref[i,-1] != 0:
            lhs += "("+str(A_aug_rref[i,-1])+")"
        else:
            lhs = lhs[:-1]
        
        if lhs=="":
            lhs = "0"

        equalities.append([name, lhs])

    
    _data = problem.data
    #We do not drop the values of the eliminated variables.
    #drop = []           
    #if _data is not None:
    #    for col in _data.columns:
    #        if col not in _inputs.names and col not in problem.outputs.names:
    #            drop.append(col)
    #    _data = _data.drop(columns=drop)

    
    _models = problem.models
    #We ignorie the models attribute for now
    #if _models is not None:
    #    pass

    #TODO
    _f = None
    if 'f' in list(vars(problem).keys()):
        _f = problem.f
        if _f is not None:
            pass

    _problem = ReducedProblem(
        inputs = _inputs,
        outputs = problem.outputs,
        objectives = problem.objectives,
        constraints = _constraints,
        f = _f,                                           #TODO
        models = _models,
        data = _data,
        optima = problem.optima,
        name = problem.name,
        equalities = equalities
    )

    return _problem


def augment_data(data: pd.DataFrame, equalities: List[str], names: List[str] =None):
    """Computes augmented DataFrame based on dependencies givern by a set of equalities.

    Args:
        data (DataFrame): data to be augmented.
        equalities (List[str]): Set of equalities used for the augmentation
        names (List[str]): name of all columns given in a certain order to determine the 
        order of the columns of the returned data.

    Returns:
        A DataFrame with additional columns (augmented data)
    """
    for lhs, rhs in equalities:
        data[lhs] = data.eval(rhs)

    if names is not None:
        data = data[names]
    
    return data




import opti

problem = opti.Problem(
    inputs=[
        opti.Continuous("wLAS", [0, 0.6]),
        opti.Continuous("wFAEOS", [0, 0.75]),
        opti.Continuous("wNIO", [0, 0.85]),
        opti.Continuous("TEMPERATURE", [15, 40]),
#        opti.Categorical("KATEGORIE", domain=["A", "B"]),
        opti.Continuous("Detergent_AMOUNT", [2, 5]),
        opti.Continuous("Log_Protease_ppm", [0, np.log(3 + 1)]),
        opti.Continuous("Log_TRILON_M_LIQUID_ppm", [0, np.log(300 + 1)]),
        opti.Continuous("Log_SOKALAN_HP_20_ppm", [0, np.log(150 + 1)]),
        opti.Continuous("Log_SODIUM_CITRATE_ppm", [0, np.log(250 + 1)]),
    ],
    outputs=[
        opti.Continuous("AISE stains"),
        opti.Continuous("Protease sensitive stains"),
        opti.Continuous("Particulate soil stains"),
        opti.Continuous("Bleachable stains"),
        opti.Continuous("Cost"),
    ],
    objectives=[
        opti.Maximize("AISE stains"),
        opti.Maximize("Protease sensitive stains"),
        opti.Maximize("Particulate soil stains"),
        opti.Maximize("Bleachable stains"),
        opti.Minimize("Cost"),
    ],
    constraints=[
        opti.LinearEquality(["wLAS", "wFAEOS", "wNIO"], rhs=1),
#        opti.LinearEquality(["wFAEOS", "wNIO"], rhs=1),
#        opti.LinearEquality(["wLAS"], rhs=0)
    ],
)

print(problem)
_problem = reduce(problem)
print(_problem)



"""
from opti.problem import read_json

problem = read_json("examples/bread.json")
_problem = reduce(problem)

cols = problem.data.columns

print(_problem.data)
print(augment_data(_problem.data, _problem.equalities, names=cols))
print(problem.data)
"""
