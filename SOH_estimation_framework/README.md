### SOH Estimation Training and Testing

The SOH estimation framework uses a one-step training and testing process (using LFP as an example):
```bash
cd SOH_estimation_framework
python train_and_test.py
```

**Important:** Before running, ensure you have replaced the corresponding `data/` and `data_generated/` folders for the specific battery chemistry (see Data Organization section above).

**Model configuration (in the script):**

For **NCA/NCM** batteries, use:
```python
model = CNNTransformer(
    img_size=(128, 128),
    hidden_dim=32,
    num_layers=2,
    num_heads=4,
    dropout=0
)
num_epochs = 100
```

For **LFP** batteries, use:
```python
model = CNNTransformer(
    img_size=(128, 128),
    hidden_dim=64,
    num_layers=4,
    num_heads=8,
    dropout=0.02
)
num_epochs = 200
```

**Key parameters (adjustable in script):**
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 100 for NCA/NCM, 200 for LFP)
- `patience`: Early stopping patience (default: 20)
- `lr`: Learning rate (default: 0.0001)

**Output:** The script automatically performs training, validation, testing, and generates:
- `best_model.pth`: Best model checkpoint (saved in `./models/`)
- `training_history.png`: Training curves
- `test_predictions_highres.png`: High-resolution test results
- Evaluation metrics for train/val/test sets

**Note:** To run experiments for different battery chemistries, replace the `data/` and `data_generated/` folders before each run (see Data Organization section).


### SOH Estimation Testing and Output

After training, use `test_and_output.py` to evaluate the model and generate results.

**Two running modes:**

#### 1. Default Mode - Evaluate and Generate Plots
```bash
cd SOH_estimation_framework
python test_and_output.py
```

**Output:**
- `test_predictions.png`: Prediction scatter plot and error histogram
- `test_predictions_highres.png`: High-resolution scatter plot
- `test_results.txt`: Detailed evaluation metrics
- `test_true.csv`: All test predictions

#### 2. CSV Generation Mode - Process Original and Generated Data
```bash
cd SOH_estimation_framework
python test_and_output.py --generate-csv
```

**Output:** CSV files for both original test data and generated data (saved in `output_folder`)

---

**Configuration settings (modify in script before running):**

**1. Capacity to SOH conversion** (uncomment the appropriate line):
```python
# For NCA batteries:
targets = targets / 3.2    
outputs = outputs / 3.2

# For LFP batteries:
targets = targets / 1.06   
outputs = outputs / 1.06

# For NCM batteries:
targets = targets / 2   
outputs = outputs / 2
```

**2. Model configuration** (uncomment the appropriate block):
```python
# For NCA/NCM batteries:
model_config = {
    'img_size': (128, 128),
    'hidden_dim': 32,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0
}

# For LFP batteries:
model_config = {
    'img_size': (128, 128),
    'hidden_dim': 64,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.02
}
```

**3. Folder paths for CSV generation:**

The script processes generated data from four voltage fragment ranges: **low**, **middle**, **high**, and **random**.

**To process each fragment range, modify both `output_folder` and `generated_folder`, then run separately:**
```python
# For low voltage fragments:
output_folder = "./low_data_figure"
generated_folder = "./data_generated/low"
# Run: python test_and_output.py --generate-csv

# For middle voltage fragments:
output_folder = "./middle_data_figure"
generated_folder = "./data_generated/middle"
# Run: python test_and_output.py --generate-csv

# For high voltage fragments:
output_folder = "./high_data_figure"
generated_folder = "./data_generated/high"
# Run: python test_and_output.py --generate-csv

# For random voltage fragments:
output_folder = "./random_data_figure"
generated_folder = "./data_generated/random"
# Run: python test_and_output.py --generate-csv
```

**Example workflow:**

1. Set `output_folder = "./low_data_figure"` and `generated_folder = "./data_generated/low"` in the script
2. Run: `python test_and_output.py --generate-csv`
3. Results saved in `./low_data_figure/`
4. Change to `output_folder = "./middle_data_figure"` and `generated_folder = "./data_generated/middle"`
5. Run again for middle range, repeat for high and random

**Note:** 
- `truedata_folder = "./original"` remains unchanged for all runs
- Each run generates CSV files with two columns: the first column is true SOH values, the second column is predicted SOH values. `batteryXX_test_True.csv` contains predictions from original test data, while `batteryXX_test_Generated.csv` contains predictions from generated data.
