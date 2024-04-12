from typing import Callable, Dict, List

import numpy as np


class SingleExperiment:
    """
    Represents a single experiment, encapsulating the experiment's logic and data.

    Parameters:
    ----------
    name : str
        Name of the experiment.
    run_callable : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Callable that defines the logic for running the experiment.

    Attributes:
    ----------
    name : str
        Name of the experiment.
    run_callable : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Callable that defines the logic for running the experiment.

    Methods:
    ----------
    run(X, y)
        Executes the experiment using the provided data and labels.
    """

    def __init__(self, name: str, run_callable: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """Initializes a SingleExperiment instance.

        Parameters
        ----------
        name : str
            Name of the experiment.
        run_callable : Callable[[np.ndarray, np.ndarray],np.ndarray]
            Callable that defines the logic for running the experiment.
        """

        self.name: str = name
        self.run_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = run_callable

    def run(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Executes the experiment using the provided data and labels.

        Parameters
        ----------
        X : np.ndarray
            The input data for the experiment.
        y : np.ndarray
            The labels or ground truth data for the experiment.

        Returns
        -------
        experiment_output : np.ndarray
            The output from the experiment, typically predictions or computed results.
        """

        return self.run_callable(X, y)


class ExperimentComparator:
    """
    Compares multiple experiments using specified metrics, facilitating the evaluation of different approaches.

    Parameters:
    ----------
    metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]]
        Metrics to be used for comparing experiments, mapped by name to their respective callable implementations.

    Attributes:
    ----------
    experiments : List[SingleExperiment]
        List of experiments to be compared.
    metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]]
        Metrics to be used for comparing experiments, mapped by name to their respective callable implementations.
    X : np.ndarray
        Input data for the experiments.
    y : np.ndarray
        Labels or ground truth data for the experiments.

    Methods:
    ----------
    add_experiment(experiment)
        Adds an experiment to be compared.
    set_data(X, y)
        Sets the dataset to be used by all experiments in the comparison.
    run()
        Runs all experiments using the set data, evaluates them using the specified metrics, and returns the results.
    """

    def __init__(self, metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]]):
        """Initializes an ExperimentComparator instance.

        Parameters
        ----------
        metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]]
            Metrics to be used for comparing experiments, mapped by name to their respective callable implementations.
        """

        self.experiments: List[SingleExperiment] = []
        self.metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = metrics
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def add_experiment(self, experiment: SingleExperiment) -> None:
        """Adds an experiment to be compared.

        Parameters
        ----------
        experiment : SingleExperiment
            The experiment to add.
        """

        self.experiments.append(experiment)

    def set_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sets the dataset to be used by all experiments in the comparison.

        Parameters
        ----------
        X : np.ndarray
            Input data for the experiments.
        y : np.ndarray
            Labels or ground truth data for the experiments.
        """

        self.X = X
        self.y = y

    def run(self) -> Dict[str, Dict[str, float]]:
        """Executes all added experiments using the set data,
        evaluates them using the specified metrics, and returns the results.

        Returns
        -------
        experiment_result : Dict[str, Dict[str, float]]
            A dictionary where each key is an experiment name and each value is another dictionary mapping
            metric names to their computed values for that experiment.
        """

        results: Dict[str, Dict[str, float]] = {}
        for experiment in self.experiments:
            predictions: np.ndarray = experiment.run(self.X, self.y)
            experiment_results: Dict[str, float] = {metric_name: metric(self.y, predictions) for metric_name, metric in
                                                    self.metrics.items()}
            results[experiment.name] = experiment_results
        return results
