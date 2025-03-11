from dataclasses import dataclass, field
import random
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import datetime

from models import UNetResNet34
from dataset import get_train_val_test_loaders
from metrics import Metrics, loss_factory, metric_factory
from utilities.visualization import (
    save_model_samples,
    plot_metrics,
)


@dataclass
class TrainingConfig:
    dataset_path: str = "data/sample_dataset"
    experiment_folder: str = "experiments/experiment"
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    num_epochs: int = 5
    save_every_n_epochs: int = 2 
    batch_size: int = 8
    loss_type: Metrics = Metrics.JaccardDice
    # This metric is used for model selection and training scheduling
    validation_metric: Metrics = Metrics.JaccardDice
    # These metrics are computed on the validation set only for tracking
    other_validation_metrics: list[Metrics] = field(
        default_factory=lambda: [Metrics.Precision, Metrics.Recall, Metrics.F1Score]
    )

    def optimizer_factory(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def scheduler_factory(self, optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def get_experiment_path(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_folder}_{timestamp}"


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The random seed to use.

    Returns:
        None
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    config: TrainingConfig,
    init_n_seen_samples: int,
    epoch: int,
):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loss_fn (torch.nn.Module): The loss function for training.
        train_loader (DataLoader): DataLoader for the training data.
        config (TrainingConfig): Configuration for training.
        init_n_seen_samples (int): Current number of seen samples.
        epoch (int): Current epoch number.

    Returns:
        dict[int, float]: Dictionary mapping number of seen samples to loss.
        int: Total number of seen samples after this epoch.
    """
    model.train()
    n_seen_samples = 0
    training_losses: dict[int, float] = {}  # Number of seen samples -> loss
    for batch in train_loader:
        images, masks = batch["image"].to(config.device), batch["mask"].to(
            config.device
        )
        optimizer.zero_grad()

        outputs = model(images)
        loss = train_loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        n_seen_samples += config.batch_size
        training_losses[init_n_seen_samples + n_seen_samples] = loss.item()

        print(
            f"Epoch {epoch}, Seen {n_seen_samples} samples, Training Loss: {loss.item():.4f}"
        )

    return training_losses, n_seen_samples


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    metric_function: torch.nn.Module,
    config: TrainingConfig,
    metric_name: str,
):
    """
    Evaluate the model on a dataset using the specified metric.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader for the evaluation data.
        metric_function (torch.nn.Module): The metric function to evaluate the model.
        config (TrainingConfig): Configuration for evaluation.
        metric_name (str): Name of the metric being evaluated.

    Returns:
        float: The average metric value across all batches.
    """
    model.eval()
    metric_values = []
    for batch in loader:
        images, masks = batch["image"].to(config.device), batch["mask"].to(
            config.device
        )
        outputs = model(images)
        metric_value = metric_function(outputs, masks)
        metric_values.append(metric_value.item())

    average_metric = sum(metric_values) / len(metric_values)
    print(f"Average {metric_name}: {average_metric:.4f}")
    return average_metric


def save_checkpoint(
    model: torch.nn.Module, path: str, data_loader: DataLoader = None
) -> None:
    """
    Save model checkpoint and visualization samples of model input-outputs.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path where to save the checkpoint.
        data_loader (DataLoader): DataLoader to get sample inputs for visualization. If None, no samples are saved.

    Returns:
        None
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), path)

    # Save model input-output samples
    if data_loader is not None:
        checkpoint_dir = os.path.dirname(path)
        save_model_samples(
            model=model,
            data_loader=data_loader,
            save_dir=checkpoint_dir,
            checkpoint_name=None,  # No need for checkpoint name as samples are already in the checkpoint directory
        )


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    experiment_path: str,
) -> tuple[dict[int, float], dict[int, float], dict[Metrics, dict[int, float]]]:
    """
    Train the model for multiple epochs with validation and testing.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        validation_loader (DataLoader): DataLoader for the validation data.
        config (TrainingConfig): Configuration for training.
        experiment_path (str): Path to save experiment outputs.

    Returns:
        tuple: A tuple containing:
            - training_losses (dict[int, float]): Dictionary mapping number of seen samples to training loss.
            - validation_losses (dict[int, float]): Dictionary mapping number of seen samples to validation loss.
            - other_validation_metrics (dict[Metrics, dict[int, float]]): Dictionary mapping metrics to dictionaries of seen samples to metric values.
    """

    # Initialize dictionaries to store losses and metrics
    training_losses: dict[int, float] = {}  # Number of seen samples -> loss
    validation_losses: dict[int, float] = {}  # Number of seen samples -> loss
    other_validation_metrics: dict[Metrics, dict[int, float]] = {
        metric: {} for metric in config.other_validation_metrics
    }  # Metric -> (Number of seen samples -> metric value)

    # Initialize loss functions and metrics to compute
    train_loss_fn = loss_factory(config.loss_type)
    validation_loss_fn = loss_factory(config.loss_type)
    other_validation_metrics_fns = {
        metric: metric_factory(metric) for metric in config.other_validation_metrics
    }

    # Initialize optimizer and scheduler
    optimizer = config.optimizer_factory(model)
    scheduler = config.scheduler_factory(optimizer)

    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_path, exist_ok=True)

    # Track best validation metric
    best_validation_metric = float("inf")

    # Training loop
    n_seen_samples = 0
    for epoch in range(config.num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{config.num_epochs} {'='*20}")

        # Train epoch
        new_training_losses, new_n_seen_samples = train_one_epoch(
            model, optimizer, train_loss_fn, train_loader, config, n_seen_samples, epoch
        )
        n_seen_samples += new_n_seen_samples
        training_losses.update(new_training_losses)

        # Validation and scheduler step
        validation_loss = evaluate_model(
            model,
            validation_loader,
            validation_loss_fn,
            config,
            f"Validation {config.validation_metric.value}",
        )
        validation_losses[n_seen_samples] = validation_loss
        scheduler.step(validation_loss)

        # Track other validation metrics
        for metric in config.other_validation_metrics:
            other_validation_metric = evaluate_model(
                model,
                validation_loader,
                other_validation_metrics_fns[metric],
                config,
                f"Validation {metric.value}",
            )
            other_validation_metrics[metric][n_seen_samples] = other_validation_metric

        # Save periodic checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_dir = os.path.join(experiment_path, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            save_checkpoint(model, checkpoint_path, validation_loader)
            print(f"Saved periodic checkpoint to {checkpoint_dir}")

        # Save best model based on validation metric
        if validation_loss < best_validation_metric:
            best_validation_metric = validation_loss
            best_model_dir = os.path.join(experiment_path, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, "model.pt")
            save_checkpoint(model, best_model_path, validation_loader)
            print(f"New best model saved with validation metric: {validation_loss:.4f}")

    return training_losses, validation_losses, other_validation_metrics


def main() -> None:
    config = TrainingConfig()

    # Generate unique experiment path with timestamp
    experiment_path = config.get_experiment_path()

    # Create experiment folder if it doesn't exist
    os.makedirs(experiment_path, exist_ok=True)
    print(f"Experiment outputs will be saved to: {experiment_path}")

    set_random_seed(config.random_seed)

    train_loader, validation_loader, _ = get_train_val_test_loaders(
        config.dataset_path, config.batch_size
    )

    model = UNetResNet34.load_pretrained()
    model.to(config.device)

    training_losses, validation_losses, other_validation_metrics = train(
        model, train_loader, validation_loader, config, experiment_path
    )

    plot_metrics(training_losses, validation_losses, other_validation_metrics, experiment_path)

if __name__ == "__main__":
    main()
