# SnAr

This dataset reports the e-factor for a nucleophilic aromatic substitution following the SnAr mechanism. 
Individual data points encode four process parameters for a flow reactor to run the reaction, along with the measured e-factor (defined as the ratio of the mass waste to the mass of product).
The dataset includes 67 samples with four parameters and one objective.

```python
problem = opti.problems.SnAr()
```

<iframe width="1024" height="800" frameborder="0" scrolling="no" src="//plotly.com/~walzds/23.embed"></iframe>

!!! note "Reference"
    A.M. Schweidtmann, A.D. Clayton, N. Holmes, E. Bradford, R.A. Bourne, A.A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chem. Eng. J. 352 (2018) 277-282.
    [DOI](https://doi.org/10.1016/j.cej.2018.07.031).
    Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).
