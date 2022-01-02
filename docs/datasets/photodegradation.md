# Photodegradation

This dataset reports the degradation of polymer blends for organic solar cells under the exposure to light. 
Individual data points encode the ratios of individual polymers in one blend, along with the measured photodegradation of this blend.
The dataset includes 2,080 samples with five parameters and one objective.

```python
problem = opti.problems.Photodegradation()
```

<iframe width="1024" height="800" frameborder="0" scrolling="no" src="//plotly.com/~walzds/19.embed"></iframe>

!!! note "Reference"
    S. Langner, F. Häse, J.D. Perea, T. Stubhan, J. Hauch, L.M. Roch, T. Heumueller, A. Aspuru-Guzik, C.J. Brabec. Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multicomponent Systems. Advanced Materials, 2020, 1907801.
    [DOI](https://doi.org/10.1002/adma.201907801).
    Obtained from [Olympus](https://github.com/aspuru-guzik-group/olympus).