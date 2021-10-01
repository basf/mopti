import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from opti.constraint import Constraint, Constraints
from opti.model import LinearModel, Model, Models
from opti.objective import Minimize, Objective, Objectives, make_objective
from opti.parameter import Categorical, Continuous, Discrete, Parameter, Parameters
from opti.sampling import constrained_sampling
from opti.sampling.base import sobol_sampling

ParametersLike = Union[Parameters, List[Parameter], List[Dict]]
ObjectivesLike = Union[Objectives, List[Objective], List[Dict]]
ConstraintsLike = Union[Constraints, List[Constraint], List[Dict]]
ModelsLike = Union[Models, List[Model], List[Dict]]
PathLike = Union[str, bytes, os.PathLike]


class Problem:
    def __init__(
        self,
        inputs: ParametersLike,
        outputs: ParametersLike,
        objectives: Optional[ObjectivesLike] = None,
        constraints: Optional[ConstraintsLike] = None,
        output_constraints: Optional[ObjectivesLike] = None,
        f: Optional[Callable] = None,
        models: Optional[ModelsLike] = None,
        data: Optional[pd.DataFrame] = None,
        optima: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ):
        """An optimization problem.

        Args:
            inputs: Input parameters.
            outputs: Output parameters.
            objectives: Optimization objectives. Defaults to minimization.
            constraints: Constraints on the inputs.
            output_constraints: Constraints on the outputs.
            f: Function to evaluate the outputs for given inputs.
                Signature: f(x: np.ndarray) -> np.ndarray where both x and f(x) are 2D arrays
                for vectorized evaluation.
            data: Experimental data.
            optima: Pareto optima.
            name: Name of the problem.
        """
        self.name = name if name is not None else "Problem"
        self.inputs = inputs if isinstance(inputs, Parameters) else Parameters(inputs)
        self.outputs = (
            outputs if isinstance(outputs, Parameters) else Parameters(outputs)
        )

        if objectives is None:
            self.objectives = Objectives([Minimize(m) for m in self.outputs.names])
        elif isinstance(objectives, Objectives):
            self.objectives = objectives
        else:
            self.objectives = Objectives(objectives)

        if isinstance(constraints, Constraints):
            pass
        elif not constraints:
            constraints = None
        else:
            constraints = Constraints(constraints)
            if len(constraints) == 0:  # no valid constraints
                constraints = None
        self.constraints = constraints

        if isinstance(output_constraints, Objectives) or output_constraints is None:
            self.output_constraints = output_constraints
        else:
            self.output_constraints = Objectives(output_constraints)

        if isinstance(models, Models) or models is None:
            self.models = models
        else:
            self.models = Models(models)

        if f is not None:
            self.f = f

        # checks
        self.set_data(data)
        self.set_optima(optima)
        self.check_problem()
        self.check_models()

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "Problem(\n"
        s += f"name={self.name},\n"
        s += f"inputs={self.inputs},\n"
        s += f"outputs={self.outputs},\n"
        s += f"objectives={self.objectives},\n"
        if self.output_constraints is not None:
            s += f"output_constraints={self.output_constraints},\n"
        if self.constraints is not None:
            s += f"constraints={self.constraints},\n"
        if self.models is not None:
            s += f"models={self.models},\n"
        if self.data is not None:
            s += f"data=\n{self.data.head()}\n"
        if self.optima is not None:
            s += f"optima=\n{self.optima.head()}\n"
        return s + ")"

    @staticmethod
    def from_config(config: dict) -> "Problem":
        """Create a Problem instance from a configuration dict."""
        outputs = Parameters(config["outputs"])

        objectives = config.get("objectives", None)
        if objectives is not None:
            objectives = [make_objective(**o) for o in config["objectives"]]
        else:
            objectives = Objectives([Minimize(d.name) for d in outputs])

        output_constraints = config.get("output_constraints", None)
        if output_constraints is not None:
            output_constraints = [
                make_objective(**o) for o in config["output_constraints"]
            ]

        models = config.get("models", None)
        if models is not None:
            models = Models(models)

        data = config.get("data", None)
        if data:
            data = pd.DataFrame(**config["data"])

        optima = config.get("optima", None)
        if optima:
            optima = pd.DataFrame(**config["optima"])

        return Problem(
            inputs=config["inputs"],
            outputs=outputs,
            objectives=objectives,
            constraints=config.get("constraints", None),
            output_constraints=output_constraints,
            models=models,
            data=data,
            optima=optima,
            name=config.get("name", None),
        )

    def to_config(self) -> Dict:
        """Return json-serializable configuration dict."""

        config = {
            "name": self.name,
            "inputs": self.inputs.to_config(),
            "outputs": self.outputs.to_config(),
            "objectives": self.objectives.to_config(),
        }
        if self.output_constraints is not None:
            config["output_constraints"] = self.output_constraints.to_config()
        if self.constraints is not None:
            config["constraints"] = self.constraints.to_config()
        if self.models is not None:
            config["models"] = self.models.to_config()
        if self.data is not None:
            config["data"] = self.data.replace({np.nan: None}).to_dict("split")
        if self.optima is not None:
            config["optima"] = self.optima.replace({np.nan: None}).to_dict("split")
        return config

    @staticmethod
    def from_json(fname: PathLike) -> "Problem":
        """Read a problem from a JSON file."""
        with open(fname, "rb") as infile:
            config = json.loads(infile.read())
        return Problem.from_config(config)

    def to_json(self, fname: PathLike) -> None:
        """Save a problem from a JSON file."""
        with open(fname, "wb") as outfile:
            b = json.dumps(self.to_config(), ensure_ascii=False, indent=2)
            outfile.write(b.encode("utf-8"))

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the function on a DataFrame"""
        x = data[self.inputs.names].values
        y = self.f(x)
        if y.ndim == 1:
            y = y[:, None]
        return pd.DataFrame(y, index=data.index, columns=self.outputs.names)

    def check_problem(self) -> None:
        """Check if input and output parameters are consistent."""
        # check if inputs and outputs are consistent
        duplicates = set(self.inputs.names).intersection(self.outputs.names)
        if duplicates:
            raise ValueError(f"Parameter name in both inputs and outputs: {duplicates}")

        # check if objectives are consistent
        all_parameters = self.inputs.names + self.outputs.names
        for obj in self.objectives:
            p = obj.parameter
            if p not in all_parameters:
                raise ValueError(f"Objective refers to unknown parameter: {p}")

    def check_data(self, data: pd.DataFrame) -> None:
        """Check if data is consistent with input and output parameters."""
        for p in self.inputs + self.outputs:
            # check if parameter is present
            if p.name not in data.columns:
                raise ValueError(f"Parameter {p.name} not in data.")

            # check for non-numeric values for continuous / discrete parameters
            if isinstance(p, (Continuous, Discrete)):
                if not is_numeric_dtype(data[p.name]):
                    raise ValueError(f"Non-numeric data for parameter {p.name}.")

            # check for undefined categories for categorical parameters
            elif isinstance(p, Categorical):
                if not p.contains(data[p.name]).all():
                    raise ValueError(f"Unknown category for parameter {p.name}.")

        # inputs need to be complete
        for p in self.inputs:
            if data[p.name].isnull().any():
                raise ValueError(f"Missing values for input parameter {p.name}.")

        # outputs need to have at least one observation
        for p in self.outputs:
            if data[p.name].isnull().all():
                raise ValueError(f"No value for output parameter {p.name} in data.")

    def check_models(self) -> None:
        """Check if the models are well defined"""
        if self.models is None:
            return

        for model in self.models:
            # models need to refer to output parameters
            for n in model.names:
                if n not in self.outputs.names:
                    raise ValueError(f"Model {model} refers to unknown outputs")

            if isinstance(model, LinearModel):
                if len(model.coefficients) != self.n_inputs:
                    raise ValueError(f"Model {model} has wrong number of coefficients.")

    def set_data(self, data: Optional[pd.DataFrame]) -> None:
        """Set the data."""
        if data is not None:
            self.check_data(data)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        """Return `self.data` if it exists or an empty dataframe."""
        if self.data is None:
            return pd.DataFrame(columns=self.inputs.names + self.outputs.names)
        return self.data

    def add_data(self, data: pd.DataFrame) -> None:
        """Add a number of data points."""
        self.check_data(data)
        self.data = pd.concat([self.data, data], axis=0)

    def set_optima(self, optima: Optional[pd.DataFrame]) -> None:
        """Set the optima / Pareto front."""
        if optima is not None:
            self.check_data(optima)
        self.optima = optima

    def get_X(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return the input values in `data` or `self.data`."""
        if data is not None:
            return data[self.inputs.names].values
        return self.get_data()[self.inputs.names].values

    def get_Y(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Return the output values in `data` or `self.data`."""
        if data is not None:
            return data[self.outputs.names].values
        return self.get_data()[self.outputs.names].values

    def get_XY(
        self,
        outputs: Optional[List[str]] = None,
        data: Optional[pd.DataFrame] = None,
        continuous: str = "none",
        discrete: str = "none",
        categorical: str = "none",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the input and output values as numeric numpy arrays.

        Rows with missing output values will be dropped.
        Input values are assumed to be complete.
        Categorical outputs are one-hot or label encoded.

        Args:
            outputs (optional): Subset of the outputs to consider.
            data (optional): Dataframe to consider instead of problem.data
        """
        if outputs is None:
            outputs = self.outputs.names
        if data is None:
            data = self.get_data()
        notna = data[outputs].notna().all(axis=1)
        X = self.inputs.transform(
            data, continuous=continuous, discrete=discrete, categorical=categorical
        )[notna].values
        Y = data[outputs][notna].values
        return X, Y

    def get_X_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper data bounds."""
        X = self.get_X()
        xlo = X.min(axis=0)
        xhi = X.max(axis=0)
        b = xlo == xhi
        xhi[b] = xlo[b] + 1  # prevent division by zero when dividing by (xhi - xlo)
        return xlo, xhi

    def sample_inputs(self, n_samples=10) -> pd.DataFrame:
        """Uniformly sample points from the input space subject to the constraints."""
        if self.constraints is None:
            return sobol_sampling(n_samples, self.inputs)
        return constrained_sampling(n_samples, self.inputs, self.constraints)

    def create_initial_data(self, n_samples: int = 10) -> None:
        """Create an initial data set by sampling uniformly from the input space and
        evaluating f(x) at the sampled inputs.
        """
        X = self.sample_inputs(n_samples)
        self.data = pd.concat([X, self.eval(X)], axis=1)


def read_json(filepath: PathLike) -> Problem:
    """Read a problem specification from a JSON file."""
    return Problem.from_json(filepath)
