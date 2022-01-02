# Benzylation

This dataset reports the yield of undesired product (impurity) in an N-benzylation reaction. 
Four conditions of this reaction performed in a flow reactor can be controlled to minimize the yield of impurity.
The dataset includes 73 samples with four parameters and one objective.

```python
problem = opti.problems.Benzylation()
```

<iframe width="1024" height="800" frameborder="0" scrolling="no" src="//plotly.com/~walzds/14.embed"></iframe>

!!! note "Reference"
    A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.
    [DOI](https://doi.org/10.1016/j.cej.2018.07.031).
    Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).