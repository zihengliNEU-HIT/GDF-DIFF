import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.io import loadmat
import matplotlib
matplotlib.use('TkAgg')  


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def process_gaf_image(gaf_img):
    """
    Preprocess a single GAF image for CNN input.

    """
    # Cast to float32
    gaf_img = gaf_img.astype(np.float32)

    # Normalize to [0, 1]
    gaf_img = gaf_img / 255.0

    # Add channel dimension
    processed_img = np.expand_dims(gaf_img, axis=0)

    return processed_img


# Defined at top-level to avoid multiprocessing serialization issues
def collate_batch(batch):
    """
    Collate a list of samples into a mini-batch.

    """
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y


def create_battery_dataset(battery_data, capacity_range=None):
    """
    Build samples from (battery_name, GAF_cell, SOH_cell) and normalize labels.

    """
    samples = []
    all_capacities = []

    # Collect capacities across all provided batteries (for range estimation)
    for _, _, cap_vec in battery_data:
        for i in range(cap_vec.shape[0]):
            capacity_value = float(cap_vec[i])
            all_capacities.append(capacity_value)

    # Determine normalization range
    if capacity_range is None:
        capacity_min = float(min(all_capacities))
        capacity_max = float(max(all_capacities))
        capacity_range = (capacity_min, capacity_max)
    else:
        capacity_min, capacity_max = capacity_range

    print(f"Capacity range: [{capacity_min:.4f}, {capacity_max:.4f}]")

    # Convert each cycle into a sample (image, label)
    for battery_name, gaf_imgs, cap_vec in battery_data:
        n_cycles = gaf_imgs.shape[0]

        for i in range(n_cycles):
            # Extract one cycle GAF image from MATLAB cell array
            gaf_img = gaf_imgs[i, 0]

            # Extract capacity label (float)
            capacity = float(cap_vec[i])

            # Normalize capacity label
            normalized_capacity = (capacity - capacity_min) / (capacity_max - capacity_min)

            # Preprocess GAF image to [1, H, W]
            processed_img = process_gaf_image(gaf_img)

            # Convert to torch tensors
            img_tensor = torch.tensor(processed_img, dtype=torch.float32)
            capacity_tensor = torch.tensor(normalized_capacity, dtype=torch.float32)

            # Store sample
            samples.append((img_tensor, capacity_tensor))

    return samples, capacity_range


def get_dataloader(samples, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the provided samples.

    """
    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def load_battery_gaf_data(data_dir, batch_size=32, num_workers=0):
    """
    Load battery GAF datasets from .mat files and build DataLoaders.

    """
    # Identify dataset split files
    train_files = [f for f in os.listdir(data_dir) if 'train_GAF.mat' in f]
    val_files = [f for f in os.listdir(data_dir) if 'val_GAF.mat' in f]
    test_files = [f for f in os.listdir(data_dir) if 'test_GAF.mat' in f]

    print(
        f"Found {len(train_files)} train files, "
        f"{len(val_files)} val files, "
        f"{len(test_files)} test files"
    )

    # -----------------------------
    # Load training split
    # -----------------------------
    train_data = []
    for mat_file in train_files:
        file_path = os.path.join(data_dir, mat_file)

        # Load MATLAB file
        data = loadmat(file_path)

        # Extract GAF images (cell array)
        gaf_imgs = data['GAF_cell']

        # Extract capacity/SOH vector
        cap_vec = data['SOH_cell']

        # Store split record: (identifier, images, labels)
        train_data.append((mat_file, gaf_imgs, cap_vec))

        # Logging
        print(f"Loaded train file {mat_file}, number of cycles: {gaf_imgs.shape[0]}")

    # -----------------------------
    # Load validation split
    # -----------------------------
    val_data = []
    for mat_file in val_files:
        file_path = os.path.join(data_dir, mat_file)

        # Load MATLAB file
        data = loadmat(file_path)

        # Extract GAF images (cell array)
        gaf_imgs = data['GAF_cell']

        # Extract capacity/SOH vector
        cap_vec = data['SOH_cell']

        # Store split record: (identifier, images, labels)
        val_data.append((mat_file, gaf_imgs, cap_vec))

        # Logging
        print(f"Loaded val file {mat_file}, number of cycles: {gaf_imgs.shape[0]}")

    # -----------------------------
    # Load test split
    # -----------------------------
    test_data = []
    for mat_file in test_files:
        file_path = os.path.join(data_dir, mat_file)

        # Load MATLAB file
        data = loadmat(file_path)

        # Extract GAF images (cell array)
        gaf_imgs = data['GAF_cell']

        # Extract capacity/SOH vector
        cap_vec = data['SOH_cell']

        # Store split record: (identifier, images, labels)
        test_data.append((mat_file, gaf_imgs, cap_vec))

        # Logging
        print(f"Loaded test file {mat_file}, number of cycles: {gaf_imgs.shape[0]}")

    # -----------------------------
    # Build datasets (train defines capacity_range)
    # -----------------------------
    train_samples, capacity_range = create_battery_dataset(train_data)

    # Use training capacity_range for val/test
    val_samples, _ = create_battery_dataset(val_data, capacity_range)
    test_samples, _ = create_battery_dataset(test_data, capacity_range)

    print(
        f"Dataset sizes — "
        f"Train: {len(train_samples)}, "
        f"Val: {len(val_samples)}, "
        f"Test: {len(test_samples)}"
    )

    # -----------------------------
    # Build DataLoaders
    # -----------------------------
    train_loader = get_dataloader(train_samples, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_dataloader(val_samples, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = get_dataloader(test_samples, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, capacity_range


def visualize_batch(dataloader, capacity_range=None, num_samples=8):
    """
    Visualize a batch of GAF images with associated capacity labels.

    """
    for images, capacities in dataloader:
        # Number of samples to show
        n = min(num_samples, images.shape[0])

        # Create figure grid
        fig, axes = plt.subplots(2, n // 2, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(n):
            # Extract image (remove channel dimension)
            img = images[i, 0].numpy()

            # Extract normalized label
            capacity_norm = capacities[i].item()

            # De-normalize label if range is provided
            if capacity_range:
                capacity_min, capacity_max = capacity_range
                capacity_original = capacity_norm * (capacity_max - capacity_min) + capacity_min
                title = f"Capacity: {capacity_original:.4f} (Norm: {capacity_norm:.2f})"
            else:
                title = f"Normalized capacity: {capacity_norm:.4f}"

            # Plot image
            im = axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(title)
            axes[i].axis('off')

        # Shared colorbar
        plt.colorbar(im, ax=axes, shrink=0.8, label='Normalized pixel value')

        plt.tight_layout()
        plt.show()
        break


if __name__ == "__main__":
    data_dir = "./data"

    # Single-process loading by default
    train_loader, val_loader, test_loader, capacity_range = load_battery_gaf_data(
        data_dir, batch_size=32, num_workers=0
    )

    # Inspect one batch
    for x, y in train_loader:
        print(f"Input shape: {x.shape}, Label shape: {y.shape}")
        break

    # Visualize samples
    print("Visualizing training samples...")
    visualize_batch(train_loader, capacity_range, num_samples=8)
