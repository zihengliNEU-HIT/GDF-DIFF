import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torchvision.utils import make_grid


def save_training_samples(inputs, targets, generated, save_path, num_samples=4):
    """
    Save training samples showing input fragments, targets, and generated images.
    
    Args:
        inputs: Input condition images of shape [batch_size, 1, height, width]
        targets: Target images of shape [batch_size, 1, height, width]
        generated: Generated images of shape [batch_size, 1, height, width]
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(num_samples, inputs.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        axes[i, 0].imshow(inputs[i, 0], cmap='gray')
        axes[i, 0].set_title("FRAG")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(targets[i, 0], cmap='gray')
        axes[i, 1].set_title("Target")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(generated[i, 0], cmap='gray')
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Training samples saved to {save_path}")


def save_comparison_grid(inputs, targets, generated, save_path, num_samples=16):
    """
    Save comparison grid showing multiple samples in a grid layout.
    
    Args:
        inputs: Input condition images of shape [batch_size, 1, height, width]
        targets: Target images of shape [batch_size, 1, height, width]
        generated: Generated images of shape [batch_size, 1, height, width]
        save_path: Path to save the grid visualization
        num_samples: Number of samples to include in grid
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(num_samples, len(inputs))
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        if i < n:
            inner_fig = plt.figure(figsize=(9, 3))
            inner_axes = [inner_fig.add_subplot(1, 3, j+1) for j in range(3)]

            inner_axes[0].imshow(inputs[i][0], cmap='gray')
            inner_axes[0].set_title("FRAG")
            inner_axes[0].axis('off')

            inner_axes[1].imshow(targets[i], cmap='gray')
            inner_axes[1].set_title("Target")
            inner_axes[1].axis('off')

            inner_axes[2].imshow(generated[i], cmap='gray')
            inner_axes[2].set_title("Generated")
            inner_axes[2].axis('off')

            plt.tight_layout()
            temp_path = os.path.join(os.path.dirname(save_path), f"temp_sample_{i}.png")
            inner_fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(inner_fig)

            img = plt.imread(temp_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Sample {i+1}")
            axes[i].axis('off')
            os.remove(temp_path)
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison grid saved to {save_path}")