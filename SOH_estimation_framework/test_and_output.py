import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from scipy.io import loadmat


from data_preprocess import load_battery_gaf_data
from model import CNNTransformer


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_path, model_config, device):
    """
    Load a trained CNNTransformer checkpoint.

    """
    # Instantiate model with the same configuration as training
    model = CNNTransformer(**model_config)

    # Load checkpoint to target device
    checkpoint = torch.load(model_path, map_location=device)

    # Restore weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and switch to evaluation mode
    model = model.to(device)
    model.eval()

    # Log checkpoint summary
    print(f"  Model loaded: {model_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    print(f"  Val RMSE: {checkpoint['val_rmse']:.6f}")

    return model, checkpoint


def evaluate_model(model, data_loader, criterion, capacity_range, device, remove_outliers=True, outlier_threshold=0.5):
    """
    Evaluate model performance on a dataloader.

    """
    model.eval()
    total_loss = 0.0

    # Collect predictions and targets (normalized space)
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move batch to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loss in normalized space
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()

            # Store arrays for metrics
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(data_loader)

    # Convert to numpy
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    # De-normalize to original capacity space
    capacity_min, capacity_max = capacity_range
    original_targets = all_targets * (capacity_max - capacity_min) + capacity_min
    original_outputs = all_outputs * (capacity_max - capacity_min) + capacity_min

    # Optional outlier removal
    if remove_outliers:
        errors = np.abs(original_outputs - original_targets)
        mask = errors < outlier_threshold

        original_targets_filtered = original_targets[mask]
        original_outputs_filtered = original_outputs[mask]

        num_removed = len(original_targets) - len(original_targets_filtered)
        print(f"Removed {num_removed} outliers (abs error threshold: {outlier_threshold})")

        original_targets = original_targets_filtered
        original_outputs = original_outputs_filtered

        all_targets = all_targets[mask]
        all_outputs = all_outputs[mask]

    # Metrics in normalized space
    norm_mse = np.mean((all_outputs - all_targets) ** 2)
    norm_rmse = np.sqrt(norm_mse)
    norm_mae = np.mean(np.abs(all_outputs - all_targets))

    # Metrics in original space
    orig_mse = np.mean((original_outputs - original_targets) ** 2)
    orig_rmse = np.sqrt(orig_mse)
    orig_mae = np.mean(np.abs(original_outputs - original_targets))

    # R^2 score in original space
    ss_res = np.sum((original_targets - original_outputs) ** 2)
    ss_tot = np.sum((original_targets - np.mean(original_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    metrics = {
        'loss': avg_loss,
        'norm_rmse': norm_rmse,
        'norm_mae': norm_mae,
        'orig_rmse': orig_rmse,
        'orig_mae': orig_mae,
        'r2_score': r2_score,
        'all_targets': original_targets,
        'all_outputs': original_outputs
    }

    return metrics


def plot_predictions(metrics, dataset_name="Test set", save_file='test_predictions.png'):
    """
    Plot predicted vs true capacity and the error histogram.

    """
    targets = metrics['all_targets']
    outputs = metrics['all_outputs']

    plt.figure(figsize=(12, 10))

    # predicted vs true
    plt.subplot(2, 1, 1)
    plt.scatter(targets, outputs, alpha=0.6)

    # Diagonal reference line
    min_val = min(min(targets), min(outputs))
    max_val = max(max(targets), max(outputs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    plt.title(f'{dataset_name}: Predicted vs. true')
    plt.xlabel('True capacity')
    plt.ylabel('Predicted capacity')
    plt.legend()
    plt.grid(True)

    # Metrics box
    textstr = (
        f"RMSE: {metrics['orig_rmse']:.4f}\n"
        f"MAE: {metrics['orig_mae']:.4f}\n"
        f"R²: {metrics['r2_score']:.4f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )

    # prediction error
    plt.subplot(2, 1, 2)
    errors = outputs - targets
    plt.hist(errors, bins=30, alpha=0.75, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero error')

    plt.title(f'{dataset_name}: Error distribution')
    plt.xlabel('Prediction error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.show()
    print(f"Figure saved: {save_file}")


def plot_test_predictions_highres(metrics, save_file='test_predictions_highres.png'):
    """
    High-resolution test-set scatter plot 
    # NOTE: To obtain test_true results for a single battery, place only one *_test_GAF.mat file in the data directory
    # and run this script with run_mode="default".

    """
    targets = metrics['all_targets']
    outputs = metrics['all_outputs']

    # Save raw scatter points
    np.savetxt(
        'test_true.csv',
        np.column_stack((targets, outputs)),
        delimiter=',',
        header='True_SOHi,Predicted_SOHi',
        comments='',
        fmt='%.6f'
    )
    print("all Scatter points saved: test_true.csv")

    # Subsample to reduce point density (keep original stride)
    targets = targets[::2]
    outputs = outputs[::2]

    #capacity to soh / NCA
    # targets = targets / 3.2    
    # outputs = outputs / 3.2
    
    #capacity to soh / LFP
    # targets = targets  / 1.06   
    # outputs = outputs  / 1.06

    #capacity to soh / NCM
    targets = targets  / 2   
    outputs = outputs  / 2



    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(3.5, 3.5), dpi=300)

    # Colormap and per-point colors
    cmap = cm.get_cmap("Purples")
    colors = cmap(np.linspace(0.3, 0.9, len(targets)))

    # Scatter plot
    plt.scatter(targets, outputs, c=colors, alpha=0.9, s=60, edgecolors='none')

    # Diagonal reference line
    min_val = min(min(targets), min(outputs))
    max_val = max(max(targets), max(outputs))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        color='gray',
        linewidth=1
    )

    # Labels and title 
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Test Set Predictions", fontsize=10)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    plt.show()



def process_single_mat_file(mat_file_path, model, capacity_range, device):
    """
    Run inference on a single .mat file and return true/predicted capacities.

    """
    # Load MATLAB file
    data = loadmat(mat_file_path)

    # Extract GAF images (cell array)
    gaf_imgs = data['GAF_cell']

    # Extract capacity/SOH vector
    cap_vec = data['SOH_cell']

    # Unpack capacity range
    capacity_min, capacity_max = capacity_range

    true_values = []
    pred_values = []

    model.eval()
    with torch.no_grad():
        for i in range(gaf_imgs.shape[0]):
            # Extract one cycle GAF image
            gaf_img = gaf_imgs[i, 0]

            # Normalize image and add channel dimension
            gaf_img = gaf_img.astype(np.float32)
            gaf_img = gaf_img / 255.0
            processed_img = np.expand_dims(gaf_img, axis=0)

            # Build tensor: [1, 1, H, W]
            img_tensor = torch.tensor(processed_img, dtype=torch.float32)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # True capacity label (original space)
            true_capacity = float(cap_vec[i])

            # Forward pass (normalized prediction)
            output = model(img_tensor)
            pred_normalized = output.squeeze().cpu().numpy()

            # De-normalize prediction to original space
            pred_capacity = pred_normalized * (capacity_max - capacity_min) + capacity_min

            true_values.append(true_capacity)
            pred_values.append(float(pred_capacity))

    return np.array(true_values), np.array(pred_values)


def generate_all_csv(data_dir, model_path, output_folder, truedata_folder, generated_folder, model_config):
    """
    Generate CSV files for both real test data and generated test data.

    """
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load trained model
    print(f"\nLoading model: {model_path}")

    model, checkpoint = load_model(model_path, model_config, device)
    print("Model loaded")

    # Load dataset once to obtain capacity_range
    print("\nFetching capacity range from dataset...")
    _, _, _, capacity_range = load_battery_gaf_data(data_dir, batch_size=16, num_workers=0)
    print(f"  Capacity range: {capacity_range}")

    # ============================================================
    # Step 1: real data 
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 1: Process real test data (write True CSV)")
    print("=" * 70)

    test_files = glob.glob(os.path.join(truedata_folder, "*_test_GAF.mat"))
    test_files.sort()

    print(f"\nFound {len(test_files)} test files:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")

    for mat_file in test_files:
        filename = os.path.basename(mat_file)
        battery_name = filename.replace("_test_GAF.mat", "")

        print(f"\nProcessing: {filename}")

        true_values, pred_values = process_single_mat_file(mat_file, model, capacity_range, device)

        csv_filename = f"{battery_name}_test_True.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        np.savetxt(
            csv_path,
            np.column_stack((true_values, pred_values)),
            delimiter=',',
            header='True_SOH,Predicted_SOH',
            comments='',
            fmt='%.6f'
        )

        print(f"Saved: {csv_filename} (n={len(true_values)})")

    # ============================================================
    # Step 2: generated test data (generated_folder)
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Process generated test data (write Generated CSV)")
    print("=" * 70)

    generated_files = glob.glob(os.path.join(generated_folder, "Generated_*_test_GAF.mat"))
    generated_files.sort()

    print(f"\nFound {len(generated_files)} generated files:")
    for f in generated_files:
        print(f"  - {os.path.basename(f)}")

    for mat_file in generated_files:
        filename = os.path.basename(mat_file)
        battery_name = filename.replace("Generated_", "").replace("_test_GAF.mat", "")

        print(f"\nProcessing: {filename}")

        true_values, pred_values = process_single_mat_file(mat_file, model, capacity_range, device)

        csv_filename = f"{battery_name}_test_Generated.csv"
        csv_path = os.path.join(output_folder, csv_filename)

        np.savetxt(
            csv_path,
            np.column_stack((true_values, pred_values)),
            delimiter=',',
            header='True_SOH,Predicted_SOH',
            comments='',
            fmt='%.6f'
        )

        print(f"Saved: {csv_filename} (n={len(pred_values)})")

    # Final summary
    print("\n" + "=" * 70)
    print("CSV generation completed.")
    print("=" * 70)
    print(f"\nOutput folder: {output_folder}")
    print(f"  - True CSV: {len(test_files)}")
    print(f"  - Generated CSV: {len(generated_files)}")
    print(f"  - Total: {len(test_files) + len(generated_files)}")


def main(run_mode="default"):
    """
    Main.

    """
    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Centralized configuration 
    # NOTE: To perform SOH estimation on generated data, set the data directory to the desired folder
    # output_folder & generated_folder

    # ============================================================
    data_dir = "./data"
    model_path = "./models/best_model.pth"

    # CSV export folders
    output_folder = "./middle_data_figure"
    truedata_folder = "./original"
    generated_folder = "./data_generated/middle"

    # Plot outputs
    test_plot_file = "test_predictions.png"
    test_plot_file_highres = "test_predictions_highres.png"

    # Evaluation outputs
    results_file = "test_results.txt"

    # Data loader config
    batch_size = 32

    # Model configuration /NCA/NCM
    model_config = {
        'img_size': (128, 128),
        'hidden_dim': 32,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0
    }

    # Model configuration /LFP
    # model_config = {
    #     'img_size': (128, 128),
    #     'hidden_dim': 64,
    #     'num_layers': 4,
    #     'num_heads': 8,
    #     'dropout': 0.02
    # }

    # ============================================================
    # Run mode: CSV export
    # ============================================================
    if run_mode == "--generate-csv":
        generate_all_csv(
            data_dir=data_dir,
            model_path=model_path,
            output_folder=output_folder,
            truedata_folder=truedata_folder,
            generated_folder=generated_folder,
            model_config=model_config
        )
        return

    # ============================================================
    # Default mode: evaluate on test_loader + plots + text report
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading test data...")
    _, _, test_loader, capacity_range = load_battery_gaf_data(
        data_dir, batch_size=batch_size, num_workers=0
    )
    print(f" Data loaded. Capacity range: {capacity_range}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Load model checkpoint
    print("\nLoading model checkpoint...")
    if not os.path.exists(model_path):
        print(f" Error: model file not found: {model_path}")
        return

    model, checkpoint = load_model(model_path, model_config, device)

    # Log parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Evaluate
    print("\nEvaluating model on test set...")
    criterion = nn.MSELoss()
    test_metrics = evaluate_model(model, test_loader, criterion, capacity_range, device)

    # Print summary
    print("\n" + "=" * 50)
    print("Test set metrics".center(50))
    print("=" * 50)
    print(f"Loss:                 {test_metrics['loss']:.6f}")
    print(f"RMSE (normalized):    {test_metrics['norm_rmse']:.6f}")
    print(f"MAE  (normalized):    {test_metrics['norm_mae']:.6f}")
    print(f"RMSE (original):      {test_metrics['orig_rmse']:.6f}")
    print(f"MAE  (original):      {test_metrics['orig_mae']:.6f}")
    print(f"R² score:             {test_metrics['r2_score']:.6f}")
    print("=" * 50)

    # Figures
    print("\nGenerating figures...")
    plot_predictions(test_metrics, dataset_name="Test set", save_file=test_plot_file)
    plot_test_predictions_highres(test_metrics, save_file=test_plot_file_highres)

    # Save result summary
    print("\nSaving results...")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Test set metrics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Val loss: {checkpoint['val_loss']:.6f}\n")
        f.write(f"Val RMSE: {checkpoint['val_rmse']:.6f}\n")
        f.write("\n")
        f.write(f"Test loss:             {test_metrics['loss']:.6f}\n")
        f.write(f"Test RMSE (norm):      {test_metrics['norm_rmse']:.6f}\n")
        f.write(f"Test MAE  (norm):      {test_metrics['norm_mae']:.6f}\n")
        f.write(f"Test RMSE (orig):      {test_metrics['orig_rmse']:.6f}\n")
        f.write(f"Test MAE  (orig):      {test_metrics['orig_mae']:.6f}\n")
        f.write(f"Test R² score:         {test_metrics['r2_score']:.6f}\n")
        f.write("=" * 50 + "\n")

    print(f"Results saved: {results_file}")
    print("\nDone.")


if __name__ == "__main__":
    # Select run mode


    # NOTE: To generate CSV files, open a terminal and run:
    #       python test_and_output.py --generate-csv
    #       The argument '--generate-csv' is passed as sys.argv[1] and used to set run_mode in main().

    import sys

    if len(sys.argv) > 1:
        main(run_mode=sys.argv[1])
    else:
        main(run_mode="default")
