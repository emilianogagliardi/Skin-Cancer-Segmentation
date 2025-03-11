# Skin Cancer Segmentation Sample Project

This repository contains a sample deep learning framework for skin cancer segmentation. The project demonstrates key features for model training and evaluation.

## Training Support Features

### Metrics Tracking

The training pipeline tracks various performance metrics:

- **Loss Functions**: BCE, Jaccard, Dice, and combined JaccardDice loss
- **Evaluation Metrics**: Precision, Recall, and F1 Score
- **Visualization**: Plots of training and validation metrics saved at the end of training

### Model Snapshots

The training process saves model snapshots:

- **Periodic Checkpoints**: Saved at regular intervals during training
- **Best Model Saving**: Automatically saves the model with the best validation performance
- **Timestamped Experiments**: Each training run creates a uniquely timestamped experiment folder

### Example Outputs

The training pipeline generates visual examples to help evaluate model performance:

- **Sample Visualizations**: Saves input images alongside model predictions
- **Metric Plots**: Generates plots of all tracked metrics at the end of training
- **Experiment Organization**: All outputs are saved in a structured experiment directory 

### Data Inspection

The project includes a data inspection script for dataset exploration:

- **Dataset Statistics**: Displays information about image and mask properties
- **Interactive Visualization**: Shows batches of images with their corresponding masks
- **Model Predictions**: Can visualize model outputs alongside ground truth masks