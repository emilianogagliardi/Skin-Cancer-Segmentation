import matplotlib.pyplot as plt
import argparse
from src.dataset import get_data_loader


def print_dataset_info(dataloader):
    """Print general information about the dataset"""
    # Get a sample batch to analyze data properties
    sample_batch = next(iter(dataloader))

    print("\n=== Dataset Information ===")
    print(f"Total number of images: {len(dataloader.dataset)}")
    print(f"\nImage properties:")
    print(f"- Shape: {sample_batch['image'].shape[1:]} (channels, height, width)")
    print(
        f"- Value range: [{sample_batch['image'].min():.3f}, {sample_batch['image'].max():.3f}]"
    )
    print(f"\nMask properties:")
    print(f"- Shape: {sample_batch['mask'].shape[1:]} (channels, height, width)")
    print(
        f"- Value range: [{sample_batch['mask'].min():.3f}, {sample_batch['mask'].max():.3f}]"
    )
    print("\n=== Visualization Controls ===")
    print("- Press any key to see next batch")
    print("- Press 'q' to quit")
    print("\nStarting visualization...\n")


def show_batch(batch, num_samples=4):
    """
    Display a batch of images and their corresponding masks

    Args:
        batch: Dictionary containing 'image' and 'mask' tensors
        num_samples: Number of samples to display
    """
    images = batch["image"][:num_samples]
    masks = batch["mask"][:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for idx in range(num_samples):
        # Convert tensor to numpy and transpose from (C,H,W) to (H,W,C)
        img = images[idx].numpy().transpose(1, 2, 0)
        mask = masks[idx].numpy().squeeze()

        # Normalize image for display if needed
        img = (img - img.min()) / (img.max() - img.min())

        # Plot image
        axes[0, idx].imshow(img)
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"Image {idx+1}")

        # Plot mask
        axes[1, idx].imshow(mask, cmap="gray")
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"Ground Truth {idx+1}")

    plt.tight_layout()
    return fig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize skin lesion dataset")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of samples to visualize"
    )
    args = parser.parse_args()

    # Fixed value for batch size
    batch_size = 8

    # Create dataloader
    dataloader = get_data_loader(args.data_dir, batch_size=batch_size, shuffle=True)

    # Print dataset information
    print_dataset_info(dataloader)

    # Interactive visualization
    plt.ion()  # Turn on interactive mode

    for batch_idx, batch in enumerate(dataloader):
        # Show the batch
        fig = show_batch(batch, num_samples=args.num_samples)

        # Wait for key press
        key = plt.waitforbuttonpress()

        # If 'q' is pressed, quit
        if key:
            manager = plt.get_current_fig_manager().canvas.manager
            if manager.key_press_handler_id and manager.key == "q":
                plt.close("all")
                print("\nVisualization ended by user")
                break

        # Close the current figure before showing the next batch
        plt.close(fig)

    plt.ioff()


if __name__ == "__main__":
    main()
