# Alkox

This dataset reports the biocatalytic oxidation of benzyl alcohol by a copper radical oxidase (AlkOx). 
The effects of enzyme loading, cocatalyst loading, and pH balance on both initial rate and total conversion were assayed.
The dataset includes 104 samples with four parameters and one objective.

```python
problem = opti.problems.Alkox()
```


<iframe width="1024" height="800" frameborder="0" scrolling="no" src="//plotly.com/~walzds/8.embed"></iframe>


!!! note "Reference"

    F. Häse, M. Aldeghi, R.J. Hickman, L.M. Roch, M. Christensen, E. Liles, J.E. Hein, A. Aspuru-Guzik. Olympus: a benchmarking framework for noisy optimization and experiment planning. arXiv (2020), 2010.04153.
    [DOI](https://doi.org/10.1088/2632-2153/abedc8).
    Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).