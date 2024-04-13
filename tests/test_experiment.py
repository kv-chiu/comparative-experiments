import numpy as np

from comparative_experiments.experiment import SingleExperiment, \
    ExperimentComparator  # Adjust the import path as necessary
from comparative_experiments.metrics import mse, rmse, mae, r2


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


def test_experiment_comparator_export_results():
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

    save_path = "./test_results.csv"
    comparator.export_results(save_path)
    assert "Mock Experiment" in results, "The experiment results were not exported correctly."
    assert "Dummy Metric" in results["Mock Experiment"], "The metric results were not exported correctly."
    assert results["Mock Experiment"]["Dummy Metric"] == 1.0, "The metric value was not exported correctly."

def test_experiment_comparator_export_results_with_multiple_experiments():
    # Assuming you have a metrics setup and multiple experiments that can be evaluated
    def mock_run_callable1(X, y):
        return X  # Return X as prediction for simplicity

    def mock_run_callable2(X, y):
        return X * 2  # Return X * 2 as prediction for simplicity

    def mock_run_callable3(X, y):
        return X * 3

    experiment1 = SingleExperiment(name="Mock Experiment 1", run_callable=mock_run_callable1)
    experiment2 = SingleExperiment(name="Mock Experiment 2", run_callable=mock_run_callable2)
    experiment3 = SingleExperiment(name="Mock Experiment 3", run_callable=mock_run_callable3)
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    comparator = ExperimentComparator(metrics=metrics)
    comparator.add_experiment(experiment1)
    comparator.add_experiment(experiment2)
    comparator.add_experiment(experiment3)
    comparator.set_data(X=np.array([1, 2, 3]), y=np.array([1, 2, 3]))
    results = comparator.run()

    save_path = "./test_results.csv"
    comparator.export_results(save_path)
    assert "Mock Experiment 1" in results, "The experiment results were not exported correctly."
    assert "Mock Experiment 2" in results, "The experiment results were not exported correctly."
    assert "Mock Experiment 3" in results, "The experiment results were not exported correctly."
    assert "MSE" in results["Mock Experiment 1"], "The metric results were not exported correctly."
    assert "RMSE" in results["Mock Experiment 1"], "The metric results were not exported correctly."
    assert "MAE" in results["Mock Experiment 1"], "The metric results were not exported correctly."
    assert "R2" in results["Mock Experiment 1"], "The metric results were not exported correctly."
    assert "MSE" in results["Mock Experiment 2"], "The metric results were not exported correctly."
    assert "RMSE" in results["Mock Experiment 2"], "The metric results were not exported correctly."
    assert "MAE" in results["Mock Experiment 2"], "The metric results were not exported correctly."
    assert "R2" in results["Mock Experiment 2"], "The metric results were not exported correctly."
    assert "MSE" in results["Mock Experiment 3"], "The metric results were not exported correctly."
    assert "RMSE" in results["Mock Experiment 3"], "The metric results were not exported correctly."
    assert "MAE" in results["Mock Experiment 3"], "The metric results were not exported correctly."
    assert "R2" in results["Mock Experiment 3"], "The metric results were not exported correctly."
    assert results["Mock Experiment 1"]["MSE"] == 0.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 1"]["RMSE"] == 0.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 1"]["MAE"] == 0.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 1"]["R2"] == 1.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 2"]["MSE"] == 4.666666666666667, "The metric value was not exported correctly."
    assert results["Mock Experiment 2"]["RMSE"] == 2.160246899469287, "The metric value was not exported correctly."
    assert results["Mock Experiment 2"]["MAE"] == 2.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 2"]["R2"] == -6.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 3"]["MSE"] == 18.666666666666668, "The metric value was not exported correctly."
    assert results["Mock Experiment 3"]["RMSE"] == 4.320493798938574, "The metric value was not exported correctly."
    assert results["Mock Experiment 3"]["MAE"] == 4.0, "The metric value was not exported correctly."
    assert results["Mock Experiment 3"]["R2"] == -27.0, "The metric value was not exported correctly."
