import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime


@torch.no_grad()
def save_model_samples(model, data_loader, save_dir, checkpoint_name=None):
    """
    Save sample inputs and outputs from the model for visualization.

    Args:
        model: The model to generate predictions
        data_loader: DataLoader to get sample inputs
        save_dir: Directory to save the samples
        checkpoint_name: Name of the checkpoint (used in the sample filename)
    """
    # Fixed number of samples to save
    num_samples = 5

    # Get batch size from dataloader
    batch_size = data_loader.batch_size

    model.eval()

    # Create a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create samples directory if it doesn't exist
    samples_dir = save_dir
    os.makedirs(samples_dir, exist_ok=True)

    # Get a batch of data
    for i, batch in enumerate(data_loader):
        if i >= (num_samples // batch_size) + 1:
            break
        device = next(model.parameters()).device
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        outputs = model(images)

        # Convert predictions to binary masks (threshold at 0.5)
        predictions = (torch.sigmoid(outputs) > 0.5).float()

        # Save each sample in the batch
        for j in range(min(images.size(0), num_samples - i * batch_size)):
            # Get the image, ground truth mask, and prediction
            image = images[j].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
            mask = masks[j].cpu().permute(1, 2, 0).numpy()
            pred = predictions[j].cpu().permute(1, 2, 0).numpy()

            # Normalize image for visualization
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Create a figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot the image, ground truth mask, and prediction
            axes[0].imshow(image)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(mask[:, :, 0], cmap="gray")
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")

            axes[2].imshow(pred[:, :, 0], cmap="gray")
            axes[2].set_title("Model Prediction")
            axes[2].axis("off")

            # Save the figure with timestamp to ensure uniqueness
            sample_path = os.path.join(
                samples_dir, f"sample_{i * batch_size + j + 1}.png"
            )
            plt.savefig(sample_path, bbox_inches="tight")
            plt.close(fig)

        if (i + 1) * batch_size >= num_samples:
            break

    print(
        f"Saved {min(num_samples, len(data_loader) * batch_size)} model input-output samples to {samples_dir}"
    )


def plot_metrics(training_losses, validation_losses, test_metrics, experiment_folder):
    """
    Plot training, validation, and test metrics and save them to the experiment folder.

    Args:
        training_losses: Dictionary mapping number of seen samples to training loss
        validation_losses: Dictionary mapping number of seen samples to validation loss
        test_metrics: Dictionary mapping metric name to dictionary of number of seen samples to metric value
        experiment_folder: Path to the experiment folder
    """
    # Create experiment folder if it doesn't exist
    os.makedirs(experiment_folder, exist_ok=True)

    def plot_metric(metric_name, metric_values):
        plt.figure()
        plt.plot(list(metric_values.keys()), list(metric_values.values()))
        plt.xlabel("Number of seen samples")
        plt.ylabel(metric_name)
        plot_path = os.path.join(experiment_folder, f"{metric_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")

    plot_metric("TrainingLoss", training_losses)
    plot_metric("ValidationLoss", validation_losses)
    for metric, values in test_metrics.items():
        plot_metric(str(metric), values)
