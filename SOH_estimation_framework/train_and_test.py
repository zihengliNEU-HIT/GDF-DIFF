import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data_preprocess import load_battery_gaf_data
from model import CNNTransformer

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, num_epochs=50, patience=10, scheduler=None,
                save_dir='./models'):
    """
    Train the model with validation monitoring and early stopping.

    - Tracks training/validation loss per epoch.
    - Saves the best checkpoint based on lowest validation loss.
    - Stops when validation loss does not improve for `patience` epochs.
    """
    # Create directory for checkpoints
    os.makedirs(save_dir, exist_ok=True)

    # Training history for later visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'lr': []
    }

    # Early stopping state
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    # Move model to device
    model = model.to(device)

    # Start timer
    start_time = time.time()

    for epoch in range(num_epochs):
        # -------------------------
        # Train phase
        # -------------------------
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move batch to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Parameter update
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Periodic logging
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        # Average train loss
        avg_train_loss = train_loss / len(train_loader)

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()
        val_loss = 0.0

        # Store all predictions/targets for RMSE
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move batch to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs.squeeze(), targets)

                # Accumulate validation loss
                val_loss += loss.item()

                # Store arrays for metrics
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.squeeze().cpu().numpy())

        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Validation RMSE (normalized space)
        rmse = np.sqrt(np.mean((np.array(all_outputs) - np.array(all_targets)) ** 2))

        # Step LR scheduler using validation loss
        if scheduler:
            scheduler.step(avg_val_loss)

        # Read current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(rmse)
        history['lr'].append(current_lr)

        # Epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val RMSE: {rmse:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # -------------------------
        # Best checkpoint tracking
        # -------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Cache best model weights in memory
            best_model_state = model.state_dict().copy()

            # Reset early stopping counter
            no_improve_epochs = 0

            # Save checkpoint to disk
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_rmse': rmse,
                },
                model_path
            )

            print("✓ New best model saved.")
        else:
            # No improvement this epoch
            no_improve_epochs += 1

        # Early stopping condition
        if no_improve_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    # End timer
    total_time = time.time() - start_time
    print(f"Training finished. Total time: {total_time//60:.0f} min {total_time%60:.0f} s")

    # Restore best weights
    model.load_state_dict(best_model_state)

    return history, model


def evaluate_model(model, data_loader, criterion, capacity_range, device):
    """
    Evaluate the model on a given split.

    Returns metrics in both normalized space and original capacity space.
    """
    model.eval()
    total_loss = 0.0

    # Collect predictions and targets
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move batch to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            # Accumulate loss
            total_loss += loss.item()

            # Store arrays for metrics
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(data_loader)

    # Convert to numpy
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    # Metrics in normalized space
    norm_mse = np.mean((all_outputs - all_targets) ** 2)
    norm_rmse = np.sqrt(norm_mse)
    norm_mae = np.mean(np.abs(all_outputs - all_targets))

    # De-normalize predictions and labels to original capacity space
    capacity_min, capacity_max = capacity_range
    original_targets = all_targets * (capacity_max - capacity_min) + capacity_min
    original_outputs = all_outputs * (capacity_max - capacity_min) + capacity_min

    # Metrics in original capacity space
    orig_mse = np.mean((original_outputs - original_targets) ** 2)
    orig_rmse = np.sqrt(orig_mse)
    orig_mae = np.mean(np.abs(original_outputs - original_targets))

    # Pack metrics
    metrics = {
        'loss': avg_loss,
        'norm_rmse': norm_rmse,
        'norm_mae': norm_mae,
        'orig_rmse': orig_rmse,
        'orig_mae': orig_mae,
        'all_targets': original_targets,
        'all_outputs': original_outputs
    }

    return metrics


def plot_training_history(history):
    """
    Plot training curves: loss, RMSE, and learning rate.
    """
    plt.figure(figsize=(15, 10))

    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Validation RMSE curve
    plt.subplot(2, 2, 2)
    plt.plot(history['val_rmse'], label='Val RMSE')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # Learning rate curve
    plt.subplot(2, 2, 3)
    plt.semilogy(history['lr'], marker='o')
    plt.title('Learning rate schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_predictions(metrics, dataset_name="Test set", save_file='prediction_results.png'):
    """
    Plot predicted vs. true capacity and the error histogram.
    """
    # Extract targets and outputs 
    targets = metrics['all_targets']
    outputs = metrics['all_outputs']

    plt.figure(figsize=(12, 10))

    # Scatter: predicted vs true
    plt.subplot(2, 1, 1)
    plt.scatter(targets, outputs, alpha=0.6)

    # Diagonal reference line
    min_val = min(min(targets), min(outputs))
    max_val = max(max(targets), max(outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{dataset_name}: Predicted vs. true')
    plt.xlabel('True capacity')
    plt.ylabel('Predicted capacity')
    plt.grid(True)

    # Metrics box
    textstr = f"RMSE: {metrics['orig_rmse']:.4f}\nMAE: {metrics['orig_mae']:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )

    # Histogram: prediction error
    plt.subplot(2, 1, 2)
    errors = outputs - targets
    plt.hist(errors, bins=30, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')

    plt.title(f'{dataset_name}: Error distribution')
    plt.xlabel('Prediction error')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_file)
    plt.show()


# Main entry
if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data configuration
    data_dir = "./data"
    batch_size = 32

    # Load data loaders and capacity normalization range
    print("Loading data...")
    train_loader, val_loader, test_loader, capacity_range = load_battery_gaf_data(
        data_dir, batch_size=batch_size, num_workers=0
    )
    print(f"Data loaded. Capacity range: {capacity_range}")

    # Build model /NCA/NCM
    model = CNNTransformer(
        img_size=(128, 128),
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0
    )

    # # Build model /LFP
    # model = CNNTransformer(
    #     img_size=(128, 128),
    #     hidden_dim=64,
    #     num_layers=4,
    #     num_heads=8,
    #     dropout=0.02
    # )

    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # LR scheduler based on validation loss plateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Train model
    print("Starting training...")
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=100, #LFP=200
        patience=20,
        scheduler=scheduler,
        save_dir='./models'
    )

    # Plot training curves
    plot_training_history(history)

    # Evaluate on train split
    print("Evaluating on train set...")
    train_metrics = evaluate_model(best_model, train_loader, criterion, capacity_range, device)

    # Evaluate on validation split
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(best_model, val_loader, criterion, capacity_range, device)

    # Evaluate on test split
    print("Evaluating on test set...")
    test_metrics = evaluate_model(best_model, test_loader, criterion, capacity_range, device)

    # Print summary metrics
    print("\n===== Train set metrics =====")
    print(f"Loss: {train_metrics['loss']:.6f}")
    print(f"RMSE (orig): {train_metrics['orig_rmse']:.6f}")
    print(f"MAE  (orig): {train_metrics['orig_mae']:.6f}")

    print("\n===== Validation set metrics =====")
    print(f"Loss: {val_metrics['loss']:.6f}")
    print(f"RMSE (orig): {val_metrics['orig_rmse']:.6f}")
    print(f"MAE  (orig): {val_metrics['orig_mae']:.6f}")

    print("\n===== Test set metrics =====")
    print(f"Loss: {test_metrics['loss']:.6f}")
    print(f"RMSE (orig): {test_metrics['orig_rmse']:.6f}")
    print(f"MAE  (orig): {test_metrics['orig_mae']:.6f}")

    # Plot predictions for each split
    plot_predictions(train_metrics, dataset_name="Train set", save_file='train_predictions.png')
    plot_predictions(val_metrics, dataset_name="Validation set", save_file='val_predictions.png')
    plot_predictions(test_metrics, dataset_name="Test set", save_file='test_predictions.png')

    print("Evaluation complete. Results saved.")


import matplotlib.cm as cm


def plot_test_predictions_highres(metrics, save_file='test_predictions_highres.png'):
    """
    High-resolution test-set scatter plot (purple gradient).

    """
    targets = metrics['all_targets']
    outputs = metrics['all_outputs']

    # Subsample points (keep the original stride as provided)
    targets = targets[::2]
    outputs = outputs[::2]

    # Figure styling
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16

    # Small figure size with high resolution
    plt.figure(figsize=(3.5, 3.5), dpi=300)

    # Colormap and per-point colors
    cmap = cm.get_cmap("Purples")
    colors = cmap(np.linspace(0.3, 0.9, len(targets)))

    # Scatter plot
    plt.scatter(targets, outputs, c=colors, alpha=0.9, s=60, edgecolors='none')

    # Diagonal reference line
    min_val = min(min(targets), min(outputs))
    max_val = max(max(targets), max(outputs))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1)

    # Labels and title (English only)
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Test Set Predictions", fontsize=10)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.show()


plot_test_predictions_highres(test_metrics, save_file='test_predictions_highres.png')
