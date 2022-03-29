# Getting started

Opti problems consist of a definition of 

* the input parameters $x \in \mathbb{X}$, 
* the output parameters $y \in \mathbb{Y}$, 
* the objectives $s(y)$ (optional), 
* the input constraints $g(x) \leq 0$ (optional), 
* the output constraints $h(y)$ (optional),
* a data set of previous function evaluations $\{x, y\}$ (optional)
* and the function $f(x)$ to be optimized (optional).

## Parameters
Input and output spaces are defined using `Parameters` objects. 
For example a mixed input space of three continuous, one discrete and one categorical parameter(s), along with an output space of continuous parameters can be defined as:

```python
import opti

inputs = opti.Parameters([
    opti.Continuous("x1", domain=[0, 1]),
    opti.Continuous("x2", domain=[0, 1]),
    opti.Continuous("x3", domain=[0, 1]),
    opti.Discrete("x4", domain=[1, 2, 5, 7.5]),
    opti.Categorical("x5", domain=["A", "B", "C"])
])

outputs = opti.Parameters([
    opti.Continuous("y1", domain=[0, None]),
    opti.Continuous("y2", domain=[None, None]),
    opti.Continuous("y3", domain=[0, 100])
])
```
Note that for some of the outputs we didn't specify bounds as we may not know them.

Individual parameters can be indexed by name.
```python
inputs["x5"]
>>> Categorical("x5", domain=["A", "B", "C"])
```
and all parameter names can retrieved with
```
inputs.names
>>> ["x1", "x2", "x3", "x4", "x5"]
```

We can sample from individual parameters, parameter spaces or parameter spaces including constraints (more on that later)
```python
x5 = inputs["x1"].sample(3)
print(x5.values)
>>> array([0.72405216, 0.14914942, 0.46051132])

X = inputs.sample(5)
print(X)
>>>      x1        x2        x3   x4 x5
0  0.760116  0.063584  0.518885  7.5  A
1  0.807928  0.496213  0.885545  1.0  C
2  0.351253  0.993993  0.340414  5.0  B
3  0.385825  0.857306  0.355267  1.0  C
4  0.191907  0.993494  0.384322  2.0  A
```

We can also check for each point in a dataframe, whether it is contained in the space.
```python
inputs.contains(X)
>>> array([ True,  True,  True,  True,  True])
```

In general all opti functions operate on dataframes and thus use the parameter name to identify corresponding column. 
Hence, a dataframe may contain additional columns and columns may be in arbitrary order.
The index of a dataframe is preserved, meaning that the returned dataframe will have the same indices as the original dataframe.


## Constraints
Input constraints are defined separately from the input space.
The following constraints are supported.

**Linear constraints** (`LinearEquality` and `LinearInequality`) are expressions of the form $\sum_i a_i x_i = b$ or $\leq b$ for equality and inequality constraints respectively.
They take a list of names of the input parameters they are operating on, a list of left-hand-side coefficients $a_i$ and a right-hand-side constant $b$.
```python
# A mixture: x1 + x2 + x3 = 1
constr1 = opti.LinearEquality(["x1", "x2", "x3"], lhs=1, rhs=1)

# x1 + 2 * x3 < 0.8
constr2 = opti.LinearInequality(["x1", "x3"], lhs=[1, 2], rhs=0.8)
```
Because of the product $a_i x_i$, linear constraints cannot operate on categorical parameters.

**Nonlinear constraints** (`NonlinearEquality` and `NonlinearInequality`) take any expression that can be evaluated by [pandas.eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html), including mathematical operators such as `sin`, `exp`, `log10` or exponentiation.
```python
# The unit circle: x1**2 + x2**2 = 1
constr3 = opti.NonlinearEquality("x1**2 + x2**2 - 1")
```
Nonlinear constraints can also operate on categorical parameters and support conditional statements.
```python
# Require x1 < 0.5 if x5 == "A"
constr4 = opti.NonlinearInequality("(x1 - 0.5) * (x5 =='A')")
```

Finally, there is a **combinatorical constraint** (`NChooseK`) to express that we only want to have $k$ out of the $n$ parameters to take positive values.
Think of a mixture, where we have long list of possible ingredients, but want to limit number of ingredients in any given recipe.
```python
# Only 2 out of 3 parameters can be greater than zero
constr5 = opti.NChooseK(["x1", "x2", "x3"], max_active=2)
```

Constraints can be grouped in a container which acts as the union constraints.
```python
constraints = opti.Constraints([constr1, constr2, constr3, constr4, constr5])
```

We can check whether a point satisfies individual constraints or the list of constraints.
```python
constr2.satisfied(X).values
>>> array([False, False, True, True, True])
```

The distance to the constraint boundary can also be evaluated for use in numerical optimization methods, where values $\leq 0$ correspond to a satisified constraint.
```python
constr2(X).values
>>> array([ 0.479001  ,  0.89347371, -0.10833372, -0.05890873, -0.22377122])
```

Opti contains a number of methods to draw random samples from constrained spaces, see the sampling reference.


## Objectives
In an optimization problem we need to define the target direction or target value individually for each output.
This is done using objectives $s_m(y_m)$ so that a mixed objective optimization becomes a minimization problem.
```python
objectives = opti.Objectives([
    opti.Minimize("y1"),
    opti.Maximize("y2"),
    opti.CloseToTarget("y3", target=7)
])
```

We can compute objective values from output values to see the objective transformation applied.
```python
Y = pd.DataFrame({
    "y1": [1, 2, 3],
    "y2": [7, 4, 5],
    "y3": [5, 6.9, 12]
})
objectives(Y)
>>> minimize_y1  maximize_y2  closetotarget_y3
0            1           -7              4.00
1            2           -4              0.01
2            3           -5             25.00
```

Objectives can also be used as output constraints. 
This is different from an objective in that we want the constraint to be satisfied and not explore possible tradeoffs.


## Problem
Finally, a problem is the combination of inputs, outputs, objectives, constraints, output_constraints, (true) function and data.

```python
problem = opti.Problem(
    inputs=inputs,
    outputs=outputs,
    constraints=constraints,
    objectives=objectives
)
```
Problems can be serialized to and from a dictionary
```python
config = problem.to_config()
problem = opti.Problem(**config)
```
or to a json file
```python
problem.to_json("problem.json")
problem = opti.read_json("problem.json")
```
