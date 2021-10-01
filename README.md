# Opti

[![Tests](https://github.com/basf/mopti/actions/workflows/main.yml/badge.svg)](https://github.com/basf/mopti/actions)
[![Docs](https://github.com/basf/mopti/actions/workflows/docs.yml/badge.svg)](https://basf.github.io/mopti/)
[![PyPI](https://img.shields.io/pypi/v/mopti.svg?color=%2334D058)](https://pypi.org/project/mopti)

Opti is a Python package for specifying problems in a number of closely related fields, including experimental design, multiobjective optimization, decision making and Bayesian optimization.

**Docs**: https://basf.github.io/mopti/ <br/>
**Code**: https://github.com/basf/mopti

### Why opti? 
Opti ...
* supports mixed continuous, discrete and categorical parameter spaces for system inputs and outputs,
* separates objectives (minimize, maximize, close-to-target) from the outputs on which they operate,
* supports different specific and generic constraints as well as black-box output constraints,
* provides sampling methods for constrained mixed variable spaces,
* json-serializes problems for use in RESTful APIs and json/bson DBs, and
* provides a range of benchmark problems for (multi-objective) optimization and Bayesian optimization.
