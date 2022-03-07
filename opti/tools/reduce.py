from logging.handlers import WatchedFileHandler
import numpy as np
import pandas as pd                                                     
from typing import Callable, Dict, List, Optional, Tuple, Union


from opti import Problem
from opti.constraint import Constraints, LinearEquality
from opti.objective import Objectives
from opti.parameter import Categorical, Parameters
from sympy import Matrix


#TODO: Wie geht man mit diskreten inputs um? --> einfach domain beibehalten?
#TODO: funktion, die das Modell f für das reduzierte problem anpasst
#TODO: Was bedeutet das attribut Models in class Problem?
#TODO: data, optima, etc einbauen


def reduce(problem: Problem) -> Problem:
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
            c = LinearEquality(names=list(names[ind]),lhs=B[i,ind],rhs=B[i,-1])
            _constraints.append(c)
        else:
            if B[i,-1] < -1e-16:
                raise Warning("There is no solution in the domain of the variables "
                "that fulfills the constraints.")
        
        ind = np.where(B[i+M-1,:-1] != 0)[0]
        if len(ind)>0:
            c = LinearEquality(names=list(names[ind]),lhs=B[i+M-1,ind],rhs=B[i+M-1,-1])
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
    drop = []
    if _data is not None:
        for col in _data.columns:
            if col not in _inputs.names and col not in problem.outputs.names:
                drop.append(col)
        _data = _data.drop(columns=drop)

    #TESTEN / TODO
    _models = problem.models
    if _models is not None:
        pass

    #TESTEN / TODO
    _f = None
    if 'f' in list(vars(problem).keys()):
        _f = problem.f
        if _f is not None:
            pass

    _problem = Problem(
        inputs = _inputs,
        outputs = problem.outputs,
        objectives = problem.objectives,
        constraints = _constraints,                       #BEI DIESEN PUNKTEN NOCH EINMAL DAVID FRAGEN:
        f = _f,                                           #Einfach daten erweitern, oder?
        models = _models,                                 #Einfach daten erweitern, oder?
        data = _data,                                     #droppe alle spalten, deren Variable eliminiert wurde, richtig so?
        optima = problem.optima,                          #es ändert sich nichts, richtig so?
        name = problem.name,
        equalities = equalities
    )

    return _problem


#TODO:
def augment():
    pass





import opti
from opti.problems import Hyperellipsoid

problem = opti.Problem(
    inputs=[
        opti.Continuous("wLAS", [0, 0.6]),
        opti.Continuous("wFAEOS", [0, 0.75]),
        opti.Continuous("wNIO", [0, 0.85]),
        opti.Continuous("TEMPERATURE", [15, 40]),
        opti.Categorical("KATEGORIE", domain=["A", "B"]),
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
        opti.LinearEquality(["wFAEOS", "wNIO"], rhs=1),
        opti.LinearEquality(["wLAS"], rhs=0)
    ],
)


from opti.problem import read_json

problem = read_json("examples/bread.json")
print(problem)

_problem = reduce(problem)
print(_problem)