from typing import Optional

import pandas as pd

from opti.problem import Problem


class Algorithm:
    """Base class for Bayesian optimization algorithms"""

    def __init__(self, problem: Problem):
        self._problem = problem
        self.model = None
        self._initialize_problem()

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        self._problem = problem
        self._initialize_problem()

    def _initialize_problem(self) -> None:
        """Problem related initializations or checks."""
        pass

    def _fit_model(self) -> None:
        """Fit a probabilistic model to the available data."""
        pass

    def _tune_model(self) -> None:
        """Tune the model parameters."""
        pass

    def copy(self, data: Optional[pd.DataFrame] = None) -> "Algorithm":
        """Creates a copy of the optimizer where the data is possibly replaced."""
        new_opt = self.from_config(self.to_config())
        if data is not None:
            new_opt._problem.set_data(data)
            new_opt._fit_model()
        return new_opt

    def add_data_and_fit(self, data: pd.DataFrame) -> None:
        """Add new data points and refit the model."""
        self._problem.add_data(data)
        self._fit_model()
        self._tune_model()

    def sample_initial_data(self, n_samples: int):
        """Create an initial data set for problems with known function y=f(x)."""
        self._problem.create_initial_data(n_samples)
        self._fit_model()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the posterior predictive mean and standard deviation."""
        raise NotImplementedError

    def predict_pareto_front(self, **kwargs) -> pd.DataFrame:
        """Calculate a finite representation the Pareto front of the model."""
        raise NotImplementedError

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        """Propose a set of experiments according to the algorithm."""
        raise NotImplementedError

    def run(
        self, n_proposals: int = 1, n_steps: int = 10, show_progress: bool = True
    ) -> None:
        """Run the BO algorithm to optimize the problem."""
        if self.problem.f is None:
            raise ValueError(
                "The problem has no function defined. For external function evaluations use the propose() method instead."
            )

        for _ in range(n_steps):
            X = self.propose(n_proposals)
            Y = self.problem.eval(X)
            self.add_data_and_fit(pd.concat([X, Y], axis=1))

    def get_model_parameters(self) -> pd.DataFrame:
        """Get the parameters of the surrogate model."""
        raise NotImplementedError

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a config dict."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict) -> "Algorithm":
        """Create an algorithm instance from a configuration dict."""
        problem = Problem.from_config(config["problem"])
        parameters = config.get("parameters", {})
        return cls(problem, **parameters)
