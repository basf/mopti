# Problem reduction

When describing physical experiments there are often linear equality constraints to be considered.
For example in a formulation all ingredients of a mixture add up to 1. 

```python
problem = opti.Problem(
    inputs=[
        opti.Continuous("x1", [0.1, 1]),
        opti.Continuous("x2", [0, 0.8]),
        opti.Continuous("x3", [0.3, 0.9]),
    ],
    outputs=[opti.Continuous("y")],
    constraints=[opti.LinearEquality(["x1", "x2", "x3"], rhs=1)]
)
```

In statistical modeling linear equalities lead to multicollinearities, which makes the coefficients of linear models sensitive to noise.
For modeling tasks this collinearity can be addressed by e.g. dropping one input parameter for each corresponding equality constraint.

For sampling and optimization tasks this becomes a bit trickier as the parameter bounds and inequality constraints need to be adapted as well.
Consider in the initial example we drop $x3$ together with the linear equality. 
To ensure that solutions ($x1$, $x2$) still satisfy the box bounds and constraints, we need to add the following two inequality constraints:
$$
\begin{align}
x_3 \geq 0.3 \Longleftrightarrow x_1 + x_2 \leq 0.7 \newline
x_3 \leq 0.9 \Longleftrightarrow x_1 + x_2 \geq 0.1
\end{align}
$$

The function [`reduce_problem`](/mopti/ref-tools/#opti.tools.reduce.reduce_problem) automates this tedious task.
Given a problem containing any number of linear inequalities and at least one equality constraint, it returns an equivalent problem where the linear equalities are removed by eliminating a corresponding number of inputs.

```python
reduced_problem, transform = opti.tools.reduce_problem(problem)
print(reduced_problem)
>>> Problem(
    inputs=Parameters([
        Continuous('x2', domain=[0.0, 0.8]), 
        Continuous('x3', domain=[0.1, 1.0])
    ]),
    outputs=Parameters([Continuous('y')]),
    objectives=Objectives([Minimize('y')]),
    constraints=Constraints([
        LinearInequality(names=['x2', 'x3'], lhs=[-1.0, -1.0], rhs=-0.1),
        LinearInequality(names=['x2', 'x3'], lhs=[1.0, 1.0], rhs=0.7)
    ])
)
```

The transformer object allows to transfrom data to and from the reduced space.

```python
X = problem.sample_inputs(10)
Xr = transform.drop_data(X)
X2 = transform.augment_data(Xr)
assert np.allclose(X, X2[X.columns])
```

Equality constraints are not well supported in sampling (any form of acceptance-rejection sampling will not work) and optimization methods. 
For example population-based optimization approaches such as evolutionary algorithms only approximately support linear equalities via penalties or a conversion to two-sided inequalites.
By reducing the problem, such optimization tasks become significantly easier to solve.

Finally, let's consider a more involved example involving two mixtures, A and B, as well as an additional discrete and categorical variable, and an extra inequality constraint for some of the components of mixture A.
We also set up a function `y = f(X)` to evaluate the system.

```python
def f(X):
    y = X[["A1", "A2", "A3", "A4"]] @ [1, -2, 3, 2]
    y += X[["B1", "B2", "B3"]] @ [0.1, 0.4, 0.3]
    y += X["Temperature"] / 30
    y += X["Process"] == "process 2"
    return pd.DataFrame({"y": y})

problem = opti.Problem(
    inputs=[
        opti.Continuous("A1", [0, 0.9]),
        opti.Continuous("A2", [0, 0.8]),
        opti.Continuous("A3", [0, 0.9]),
        opti.Continuous("A4", [0, 0.9]),
        opti.Continuous("B1", [0.3, 0.9]),
        opti.Continuous("B2", [0, 0.8]),
        opti.Continuous("B3", [0.1, 1]),
        opti.Discrete("Temperature", [20, 25, 30]),
        opti.Categorical("Process", ["process 1", "process 2", "process 3"])
    ],
    outputs=[opti.Continuous("y")],
    constraints=[
        opti.LinearEquality(["A1", "A2", "A3", "A4"], rhs=1),
        opti.LinearEquality(["B1", "B2", "B3"], rhs=1),
        opti.LinearInequality(["A1", "A2"], lhs=[1, 2], rhs=0.8),
    ],
    f=f
)
```

Reducing the problem works despite the discrete and categorical inputs as these don't appear in the linear equalities.
We end up 7 out of 9 initial inputs and 5 inequality constraints, which are only referring to the remaining inputs.
```python
reduced_problem, transform = opti.tools.reduce_problem(problem)
print(reduced_problem)
>>> Problem(
    inputs=Parameters([
        Discrete('Temperature', domain=[20.0, 25.0, 30.0]),
        Categorical('Process', domain=['process 1', 'process 2', 'process 3']),
        Continuous('A2', domain=[0.0, 0.8]),
        Continuous('A3', domain=[0.0, 0.9]),
        Continuous('A4', domain=[0.0, 0.9]),
        Continuous('B2', domain=[0.0, 0.8]),
        Continuous('B3', domain=[0.1, 1.0])
    ]),
    outputs=Parameters([Continuous('y')]),
    objectives=Objectives([Minimize('y')]),
    constraints=Constraints([
        LinearInequality(names=['A2' 'A3' 'A4'], lhs=[1.0, -1.0, -1.0], rhs=-0.2),
        LinearInequality(names=['A2', 'A3', 'A4'], lhs=[-1.0, -1.0, -1.0], rhs=-0.1),
        LinearInequality(names=['A2', 'A3', 'A4'], lhs=[1.0, 1.0, 1.0], rhs=1.0),
        LinearInequality(names=['B2', 'B3'], lhs=[-1.0, -1.0], rhs=-0.1),
        LinearInequality(names=['B2', 'B3'], lhs=[1.0, 1.0], rhs=0.7)
    ])
)
```

The function `f(X)` was automaticaly wrapped so the in the reduced problem it can be evaluated for points in the reduced space, with the same result.

```
Xr = reduced_problem.sample_inputs(10)
X = transform.augment_data(Xr)
y1 = problem.f(X)
y2 = reduced_problem.f(Xr)
assert np.allclose(y1, y2)
```
