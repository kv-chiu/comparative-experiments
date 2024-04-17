import logging
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


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
        self.model = None

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

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


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

        Attributes
        ----------
        experiments : List[SingleExperiment]
            List of experiments to be compared.
        metrics : Dict[str, Callable[[np.ndarray, np.ndarray], float]]
            Metrics to be used for comparing experiments, mapped by name to their respective callable implementations.
        X : np.ndarray
            Input data for the experiments.
        y : np.ndarray
            Labels or ground truth data for the experiments.
        results : Dict[str, Dict[str, float]] | None
            A dictionary where each key is an experiment name and each value is another dictionary mapping
            metric names to their computed values for that experiment.

        Methods
        ----------
        add_experiment(experiment)
            Adds an experiment to be compared.
        set_data(X, y)
            Sets the dataset to be used by all experiments in the comparison.
        run()
            Runs all experiments using the set data, evaluates them using the specified metrics, and returns the results.
        export_results(path)
            Exports the results of the experiments to a CSV file.
        """

        self.experiments: List[SingleExperiment] = []
        self.metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = metrics
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.results: Dict[str, Dict[str, float]] | None = None

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

        logging.basicConfig(
            filename=f"{time.time()}_experiment_logs.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )
        results: Dict[str, Dict[str, float]] = {}
        for experiment in self.experiments:
            predictions: np.ndarray = experiment.run(self.X, self.y)
            experiment_results: Dict[str, float] = {
                metric_name: metric(self.y, predictions) for metric_name, metric in self.metrics.items()
            }
            results[experiment.name] = experiment_results
            # Log the results of this experiment
            logging.info(f"Experiment: {experiment.name}, Results: {experiment_results}")

        self.results = results
        return results

    def export_results(self, path: str) -> None:
        """Exports the results of the experiments to a CSV file.

        Parameters
        ----------
        path : str
            The path to save the CSV file to.
        """

        if self.results is None:
            raise ValueError("No results to export. Run the experiments first.")

        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(path)

    def get_best_experiment(self, metric_name: str, method: str = 'max') -> str:
        """Returns the name of the best experiment based on the specified metric.

        Parameters
        ----------
        metric_name : str
            The name of the metric to use for comparison.
        method : str
            The method to use for comparison ('max' or 'min').

        Returns
        -------
        best_experiment : str
            The name of the best experiment based on the specified metric.
        """

        if self.results is None:
            raise ValueError("No results to compare. Run the experiments first.")

        if method == 'max':
            best_experiment = max(self.results, key=lambda x: self.results[x][metric_name])
        elif method == 'min':
            best_experiment = min(self.results, key=lambda x: self.results[x][metric_name])
        else:
            raise ValueError("Invalid method. Please use 'max' or 'min'.")

        return best_experiment
