# Utility Scripts

This directory contains utility scripts for the skin cancer segmentation project.

## Visualization Utilities

The `visualization.py` module provides functions for visualizing model inputs, outputs, and training progress.

### Functions

- `save_model_samples(model, data_loader, save_dir, device="cpu", checkpoint_name=None)`: 
  Saves 5 sample inputs and outputs from a model for visualization.

- `create_model_progress_visualization(experiment_path, data_loader, device="cpu", num_samples=3)`:
  Creates a visualization of model progress over time by comparing predictions from different checkpoints.

- `plot_metrics(training_losses, validation_losses, test_metrics, experiment_folder)`:
  Plots training, validation, and test metrics and saves them to the experiment folder.

### Usage in Training

During training, model input-outputs are automatically saved at each checkpoint using the `save_checkpoint_and_io_samples` function, which:
1. Saves the model state to a checkpoint file
2. Saves 5 sample input-output visualizations to help track model progress

### Usage with Inspect Data Script

The `inspect_data.py` script can be used to visualize the dataset and model predictions. It now includes functionality to save model predictions and visualize model progress.

```bash
# Basic dataset visualization
python -m src.utilities.inspect_data --data_dir data/sample_dataset

# Visualize dataset with model predictions
python -m src.utilities.inspect_data --data_dir data/sample_dataset --model_path experiments/experiment_20230615_123456/checkpoints/best_model.pt

# Save model predictions
python -m src.utilities.inspect_data --data_dir data/sample_dataset --model_path experiments/experiment_20230615_123456/checkpoints/best_model.pt --save_predictions --output_dir output

# Visualize model progress
python -m src.utilities.inspect_data --data_dir data/sample_dataset --experiment_dir experiments/experiment_20230615_123456 --num_samples 5
```

## Dataset Utilities

The `split_dataset.py` script is used to split a dataset into train, validation, and test sets.

## Data Inspection

The `inspect_data.py` script provides functionality to inspect the dataset and model predictions. 