import sys
from pathlib import Path

# Add the root directory to the path to import the module
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import numpy as np
from comparative_experiments.experiment import SingleExperiment, ExperimentComparator

def main():
    # Define the metrics to be used for comparison
    metrics = {
        'Mean Squared Error': lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
        'Mean Absolute Error': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
    }

    # Initialize the comparator
    comparator = ExperimentComparator(metrics=metrics)

    # Create a simple experiment
    def mock_run_callable(X, y):
        return X  # Simple mock behavior for testing

    experiment = SingleExperiment(name="Mock Experiment", run_callable=mock_run_callable)

    # Add the experiment to the comparator
    comparator.add_experiment(experiment)

    # Set the data for the experiments
    X_test = np.array([1, 2, 3])
    y_test = np.array([2, 4, 6])
    comparator.set_data(X=X_test, y=y_test)

    # Run the experiments and evaluate them
    results = comparator.run()

    # Print the results
    for experiment_name, metrics in results.items():
        print(f"Experiment: {experiment_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

if __name__ == "__main__":
    main()
