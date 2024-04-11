import numpy as np

from comparative_experiments.experiment import SingleExperiment, \
    ExperimentComparator  # Adjust the import path as necessary


def test_single_experiment_init():
    def mock_run_callable(X, y):
        return X  # Simple mock behavior for testing

    experiment = SingleExperiment(name="Test Experiment", run_callable=mock_run_callable)
    assert experiment.name == "Test Experiment", "The experiment name was not set correctly."
    assert experiment.run_callable(np.array([1, 2, 3]),
                                   np.array([1, 2, 3])).all(), "The experiment run callable did not execute correctly."


def test_single_experiment_run():
    # Mock function for run_callable
    def mock_run_callable(X, y):
        return X * 2  # Simple mock behavior for testing

    experiment = SingleExperiment(name="Test Experiment", run_callable=mock_run_callable)
    X_test = np.array([1, 2, 3])
    y_test = np.array([2, 4, 6])

    assert np.array_equal(experiment.run(X_test, y_test),
                          X_test * 2), "The experiment run did not produce expected output."


def test_experiment_comparator_init():
    metrics = {
        'Mean Squared Error': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        'Mean Absolute Error': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    }
    comparator = ExperimentComparator(metrics=metrics)
    assert len(comparator.metrics) == 2, "The metrics were not set correctly in the comparator."


def test_experiment_comparator_add_experiment():
    metrics = {
        'Mean Squared Error': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    }
    comparator = ExperimentComparator(metrics=metrics)
    experiment = SingleExperiment(name="Mock Experiment", run_callable=lambda X, y: X)
    comparator.add_experiment(experiment)
    assert len(comparator.experiments) == 1, "The experiment was not added to the comparator correctly."


def test_experiment_comparator_set_data():
    metrics = {
        'Mean Squared Error': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
    }
    comparator = ExperimentComparator(metrics=metrics)
    X_test = np.array([1, 2, 3])
    y_test = np.array([1, 4, 9])
    comparator.set_data(X=X_test, y=y_test)
    assert np.array_equal(comparator.X, X_test), "The input data was not set correctly in the comparator."
    assert np.array_equal(comparator.y, y_test), "The labels were not set correctly in the comparator."


def test_experiment_comparator_run():
    # Assuming you have a metrics setup and a simple experiment that can be evaluated
    def mock_run_callable(X, y):
        return X  # Return X as prediction for simplicity

    experiment = SingleExperiment(name="Mock Experiment", run_callable=mock_run_callable)
    metrics = {
        'Dummy Metric': lambda y_true, y_pred: 1.0  # Dummy metric for testing
    }
    comparator = ExperimentComparator(metrics=metrics)
    comparator.add_experiment(experiment)
    comparator.set_data(X=np.array([1, 2, 3]), y=np.array([1, 2, 3]))

    results = comparator.run()
    assert results["Mock Experiment"][
               "Dummy Metric"] == 1.0, "The experiment comparator did not compute metrics correctly."
