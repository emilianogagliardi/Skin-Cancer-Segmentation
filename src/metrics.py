from enum import Enum

import torch
import segmentation_models_pytorch as smp
import numpy as np

class Metrics(Enum):
    # Loss metrics
    BCE = "BCE"
    Jaccard = "Jaccard"
    Dice = "Dice"
    JaccardDice = "JaccardDice"
    
    # Evaluation metrics
    Precision = "Precision"
    Recall = "Recall"
    F1Score = "F1Score"

def loss_factory(metric_type):
    """
    Factory function to create loss functions based on the specified type.
    
    Args:
        metric_type (Metrics): The type of loss function to create.
        
    Returns:
        callable: The loss function.
    """
    if metric_type == Metrics.BCE:
        return torch.nn.BCEWithLogitsLoss()
    elif metric_type == Metrics.Jaccard:
        return smp.losses.JaccardLoss(mode='binary')
    elif metric_type == Metrics.Dice:
        return smp.losses.DiceLoss(mode='binary')
    elif metric_type == Metrics.JaccardDice:
        return lambda x, y: smp.losses.DiceLoss(mode='binary')(x, y) + smp.losses.JaccardLoss(mode='binary')(x, y)
    else:
        raise ValueError(f"Unsupported loss type: {metric_type}. Maybe you are attempting to use a non differentiable metric as a loss function?")

def precision(mask1, mask2):
    """
    Calculate the precision between predicted and ground truth masks.
    
    Args:
        mask1 (np.ndarray): Predicted binary mask.
        mask2 (np.ndarray): Ground truth binary mask.
        
    Returns:
        float: Precision score.
    """
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate true positives and false positives
    true_positives = np.logical_and(mask1, mask2).sum()
    all_positives = mask1.sum()
    
    # Calculate precision
    precision_score = true_positives / all_positives if all_positives != 0 else 0.0
    
    return precision_score

def recall(mask1, mask2):
    """
    Calculate the recall between predicted and ground truth masks.
    
    Args:
        mask1 (np.ndarray): Predicted binary mask.
        mask2 (np.ndarray): Ground truth binary mask.
        
    Returns:
        float: Recall score.
    """
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate true positives and false negatives
    true_positives = np.logical_and(mask1, mask2).sum()
    actual_positives = mask2.sum()
    
    # Calculate recall
    recall_score = true_positives / actual_positives if actual_positives != 0 else 0.0
    
    return recall_score

def f1_score(mask1, mask2):
    """
    Calculate the F1 score between predicted and ground truth masks.
    
    Args:
        mask1 (np.ndarray): Predicted binary mask.
        mask2 (np.ndarray): Ground truth binary mask.
        
    Returns:
        float: F1 score.
    """
    # Calculate precision and recall
    prec = precision(mask1, mask2)
    rec = recall(mask1, mask2)
    
    # Calculate F1 score
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.0
    
    return f1

def metric_factory(metric_type):
    """
    Factory function to create metric functions based on the specified type.
    
    Args:
        metric_type (Metrics): The type of metric function to create.
        
    Returns:
        callable: The metric function.
    """
    if metric_type == Metrics.Precision:
        return lambda outputs, targets: torch.tensor(precision(
            (torch.sigmoid(outputs) > 0.5).cpu().numpy(), 
            targets.cpu().numpy()
        ))
    elif metric_type == Metrics.Recall:
        return lambda outputs, targets: torch.tensor(recall(
            (torch.sigmoid(outputs) > 0.5).cpu().numpy(), 
            targets.cpu().numpy()
        ))
    elif metric_type == Metrics.F1Score:
        return lambda outputs, targets: torch.tensor(f1_score(
            (torch.sigmoid(outputs) > 0.5).cpu().numpy(), 
            targets.cpu().numpy()
        ))
    # Support for loss metrics when used as evaluation metrics
    elif metric_type in [Metrics.BCE, Metrics.Jaccard, Metrics.Dice, Metrics.JaccardDice]:
        loss_fn = loss_factory(metric_type)
        return lambda outputs, targets: loss_fn(outputs, targets)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

