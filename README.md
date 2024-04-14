# Comparative Experiments

## Introduction

This project is designed to compare the performance of different algorithms on a given dataset.

### Dependencies

- Python 3.11
- Numpy

## Usage

```python
import numpy as np
from comparative_experiments.experiment import SingleExperiment, ExperimentComparator
from comparative_experiments.metrics import mse, rmse, mae, r2


def main():
    # Define the metrics to be used for comparison
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    # Initialize the comparator
    comparator = ExperimentComparator(metrics=metrics)

    # Create a simple experiment
    def mock_run_callable1(X, y):
        return X  # Simple mock behavior for testing

    def mock_run_callable2(X, y):
        return X * 2

    def mock_run_callable3(X, y):
        return X * 3

    experiment1 = SingleExperiment(name="Mock Experiment 1", run_callable=mock_run_callable1)
    experiment2 = SingleExperiment(name="Mock Experiment 2", run_callable=mock_run_callable2)
    experiment3 = SingleExperiment(name="Mock Experiment 3", run_callable=mock_run_callable3)

    # Add the experiment to the comparator
    comparator.add_experiment(experiment1)
    comparator.add_experiment(experiment2)
    comparator.add_experiment(experiment3)

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

    # Export the results to a CSV file
    save_path = "./experiment_results.csv"
    comparator.export_results(save_path)


if __name__ == "__main__":
    main()
```

Or run the example script:

```bash
python ./examples/basic_usage.py
```

### Output

|                   | MSE               | RMSE              | MAE | R2    |
|-------------------|-------------------|-------------------|-----|-------|
| Mock Experiment 1 | 4.666666666666667 | 2.160246899469287 | 2.0 | -0.75 |
| Mock Experiment 2 | 0.0               | 0.0               | 0.0 | 1.0   |
| Mock Experiment 3 | 4.666666666666667 | 2.160246899469287 | 2.0 | -0.75 |

## Test

```bash
pytest
```

## Future Work

- Impove the stability of the comparator
- [ ] Real-time Data logging
