import json
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from opti.constraint import Constraint, Constraints
from opti.model import LinearModel, Model, Models
from opti.objective import Minimize, Objective, Objectives
from opti.parameter import Categorical, Continuous, Discrete, Parameter, Parameters
from opti.sampling import constrained_sampling
from opti.sampling.base import sobol_sampling

ParametersLike = Union[Parameters, List[Parameter], List[Dict]]
ObjectivesLike = Union[Objectives, List[Objective], List[Dict]]
ConstraintsLike = Union[Constraints, List[Constraint], List[Dict]]
ModelsLike = Union[Models, List[Model], List[Dict]]
DataFrameLike = Union[pd.DataFrame, Dict]
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
        data: Optional[DataFrameLike] = None,
        optima: Optional[DataFrameLike] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """An optimization problem.

        Args:
            inputs: Input parameters.
            outputs: Output parameters.
            objectives: Optimization objectives. Defaults to minimization.
            constraints: Constraints on the inputs.
            output_constraints: Constraints on the outputs.
            f: Function to evaluate the outputs for given inputs.
                Must have the signature: f(x: pd.DataFrame) -> pd.DataFrame
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

        if isinstance(data, dict):
            data = pd.read_json(json.dumps(data), orient="split")

        if isinstance(optima, dict):
            optima = pd.read_json(json.dumps(optima), orient="split")

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

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        return 0 if self.constraints is None else len(self.constraints)

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
        return Problem(**config)

    def to_config(self) -> dict:
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
        return Problem(**config)

    def to_json(self, fname: PathLike) -> None:
        """Save a problem from a JSON file."""
        with open(fname, "wb") as outfile:
            b = json.dumps(self.to_config(), ensure_ascii=False, separators=(",", ":"))
            outfile.write(b.encode("utf-8"))

    def check_problem(self) -> None:
        """Check if input and output parameters are consistent."""
        # check for duplicate names
        duplicates = set(self.inputs.names).intersection(self.outputs.names)
        if duplicates:
            raise ValueError(f"Parameter name in both inputs and outputs: {duplicates}")

        # check if all objectives refer to an output
        for obj in self.objectives:
            if obj.name not in self.outputs.names:
                raise ValueError(f"Objective refers to unknown parameter: {obj.name}")

    def check_data(self, data: pd.DataFrame) -> None:
        """Check if data is consistent with input and output parameters."""
        for p in self.inputs + self.outputs:
            # data must contain all parameters
            if p.name not in data.columns:
                raise ValueError(
                    f"Parameter {p.name} is missing. Data must contain all parameters."
                )

            # data for continuous / discrete parameters must be numeric
            if isinstance(p, (Continuous, Discrete)):
                ok = is_numeric_dtype(data[p.name]) or data[p.name].isnull().all()
                if not ok:
                    raise ValueError(
                        f"Parameter {p.name} contains non-numeric values. Data for continuous / discrete parameters must be numeric."
                    )

            # categorical levels in data must be specified
            elif isinstance(p, Categorical):
                ok = p.contains(data[p.name]) | data[p.name].isna()
                if not ok.all():
                    unknowns = data[p.name][~ok].unique().tolist()
                    raise ValueError(
                        f"Data for parameter {p.name} contains unknown values: {unknowns}. All categorical levels must be specified."
                    )

        # inputs must be complete
        for p in self.inputs:
            if data[p.name].isnull().any():
                raise ValueError(
                    f"Input parameter {p.name} has missing data. Inputs must be complete."
                )

        # outputs must have at least one observation
        for p in self.outputs:
            if data[p.name].isnull().all():
                raise ValueError(
                    f"Output parameter {p.name} has no data. Outputs must have at least one observation."
                )

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
            for p in self.inputs:
                # Categorical levels are required to be strings. Ensure that the corresponding data is as well.
                if isinstance(p, Categorical):
                    nulls = data[p.name].isna()
                    data[p.name] = data[p.name].astype(str).mask(nulls, np.nan)

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
        """Create an initial data set for benchmark problems by sampling uniformly from the input space and evaluating f(x) at the sampled inputs."""
        if self.f is None:
            raise NotImplementedError("problem.f is not implemented for the problem.")
        X = self.sample_inputs(n_samples)
        Y = self.f(X)
        self.data = pd.concat([X, Y], axis=1)


def read_json(filepath: PathLike) -> Problem:
    """Read a problem specification from a JSON file."""
    return Problem.from_json(filepath)
