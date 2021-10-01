# Opti

![![Testing](https://github.com/basf/mopti/actions/workflows/main.yml)](https://github.com/basf/mopti/actions/workflows/main.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/mopti.svg)](https://pypi.org/project/mopti)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

Opti is a Python package for specifying problems in a number of closely related fields, including experimental design, multiobjective optimization and decision making and Bayesion optimization.

#### Why opti?
Opti 
* supports mixed continuous, discrete and categorical parameter spaces for system inputs and outputs,
* separates objectives (minimize, maximize, close-to-target) from the outputs on which they operate,
* supports different specific and generic constraints as well as black-box output constraints,
* provides sampling methods for constrained mixed variable spaces,
* json-serializes problems for use in RESTful APIs and json/bson DBs, and
* provides a range of benchmark problems for (multi-objective) optimization and Bayesian optimization.


#### Install
```
pip install mopti
```
