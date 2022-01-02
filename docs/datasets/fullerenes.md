# Fullerenes

This dataset reports the production of o-xylenyl adducts of Buckminsterfullerenes. 
Three process conditions (temperature, reaction time and ratio of sultine to C60) are varied to maximize the mole fraction of the desired product. 
Experiments are executed on a three factor fully factorial grid with six levels per factor.
The dataset includes 246 samples with three parameters and one objective.

```python
problem = opti.problems.Fullerenes()
```

<iframe width="1024" height="800" frameborder="0" scrolling="no" src="//plotly.com/~walzds/17.embed"></iframe>

!!! note "Reference"
    B.E. Walker, J.H. Bannock, A.M. Nightingale, J.C. deMello. Tuning reaction products by constrained optimisation. React. Chem. Eng., (2017), 2, 785-798. 
    [DOI](https://doi.org/10.1039/C7RE00123A).
    Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).