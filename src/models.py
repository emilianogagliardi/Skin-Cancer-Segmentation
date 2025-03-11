import torch
import torch.nn as nn
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from pathlib import Path
import os


class UNetResNet34(nn.Module):
    """
    UNet model with ResNet34 encoder backbone for binary segmentation.
    This class provides functionality to load the model from pretrained weights
    or from a local snapshot.
    """
    
    def __init__(self, pretrained=True):
        """
        Initialize the UNetResNet34 model for binary segmentation.
        
        Args:
            pretrained (bool): Whether to initialize with pretrained weights. Default is True.
        """
        super(UNetResNet34, self).__init__()
        
        # Load the model
        if pretrained:
            self.model = self.load_pretrained()
        else:
            self.model = self.create_model()
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    @staticmethod
    def create_model():
        """
        Create a UNet model with ResNet34 backbone for binary segmentation.
        
        Returns:
            nn.Module: The UNet model with ResNet34 backbone.
        """
        # Create a UNet with ResNet34 backbone using segmentation_models_pytorch
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        
        return model
    
    @staticmethod
    def load_pretrained():
        """
        Load a pretrained UNet model with ResNet34 backbone from online sources.
        
        Returns:
            nn.Module: The pretrained UNet model with ResNet34 backbone.
        """
        # Create a UNet with ResNet34 backbone and pretrained weights using segmentation_models_pytorch
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        
        return model
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        """
        Load a UNet model with ResNet34 backbone from a local checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            map_location (str or torch.device): Location to map the model to.
            
        Returns:
            UNetResNet34: The loaded model.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Create a new model instance
        model = cls(pretrained=False)
        
        # Load the state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if it exists in the state dict keys
            if all(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
        else:
            model.model.load_state_dict(checkpoint)
        
        return model
    
    def save_checkpoint(self, save_path, additional_info=None):
        """
        Save the model to a checkpoint file.
        
        Args:
            save_path (str): Path to save the checkpoint.
            additional_info (dict, optional): Additional information to save with the model.
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Prepare checkpoint
        checkpoint = {
            'state_dict': self.model.state_dict(),
        }
        
        # Add additional info if provided
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save the checkpoint
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
