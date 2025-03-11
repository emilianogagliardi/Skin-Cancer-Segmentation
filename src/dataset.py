import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms


class SkinLesionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the data directory containing 'images' and 'masks' folders
            transform (callable, optional): Optional transform to be applied to both image and mask
        """
        self.data_dir = data_dir
        self.transform = transform

        # Get list of image files
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        # List all images in the directory
        self.images = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Default transformations
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            mask = to_tensor(mask)

        return {"image": image, "mask": mask, "image_path": img_path}

    @staticmethod
    def create_dataset(data_dir, transform=None):
        """
        Creates a SkinLesionDataset with the specified transformations

        Args:
            data_dir (str): Path to the data directory
            transform: PyTorch transformations to apply

        Returns:
            SkinLesionDataset: The created dataset
        """
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )

        return SkinLesionDataset(data_dir=data_dir, transform=transform)


def get_data_loader(data_dir, batch_size=8, shuffle=True):
    """
    Creates a DataLoader for the skin lesion dataset

    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Create dataset
    dataset = SkinLesionDataset.create_dataset(data_dir)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    return dataloader


def get_train_val_test_loaders(data_dir, batch_size=8):
    """
    Creates train, validation, and test DataLoaders for the skin lesion dataset

    Args:
        data_dir (str): Path to the base data directory containing 'train', 'validation', and 'test' folders
        batch_size (int): Batch size for training

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create transform (can be customized per split if needed)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    # Create datasets for each split
    train_dataset = SkinLesionDataset.create_dataset(
        os.path.join(data_dir, "train"), transform
    )

    val_dataset = SkinLesionDataset.create_dataset(
        os.path.join(data_dir, "validation"), transform
    )

    test_dataset = SkinLesionDataset.create_dataset(
        os.path.join(data_dir, "test"), transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
