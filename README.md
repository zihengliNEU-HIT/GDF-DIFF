# GDF-DIFF
Official implementation of "Conditional Generative Diffusion Enables Ultra-Fast Battery Health Estimation from Random 100mV Charging Fragments"



### Dataset

Due to the ongoing review and publication process, the dataset is currently hosted on Zenodo and accessible via a private preview link.

Private Zenodo preview link (for reviewers and collaborators only):  
https://zenodo.org/records/18298930?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQwMDI5OGQyLTA1OTEtNDBmOS04MTUxLTExMzZhY2Y2ZWEzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJmOGQzYTI0YTgzYmViYTVmY2I2MzU0NTEyYWEzNWNiOSJ9.wNBA9Y7sGqHMNHf975LzwW3JOl8KEKOjGaBNpIz_2zCjcdGWCk8ORjTk2LFDV4PoCiXZ301pDZtoH7M3HqtXrw

The dataset will be made publicly available on Zenodo upon acceptance and formal publication, and this link will be replaced by the official public record URL and DOI.


### Data Organization

**For `Diffusion_framework/` folder:** The dataset on Zenodo has the **same directory structure** as this repository. To use the dataset:

**Simply replace the corresponding folders from Zenodo into this repository:**

- Replace `Diffusion_framework/LFP/data/` with dataset's `Diffusion_framework/LFP/data/`
- Replace `Diffusion_framework/LFP/data_for_generated/` with dataset's `Diffusion_framework/LFP/data_for_generated/`
- Replace `Diffusion_framework/LFP/generated_data/` with dataset's `Diffusion_framework/LFP/generated_data/`
- Replace `Diffusion_framework/LFP/output/` with dataset's `Diffusion_framework/LFP/output/`
- Replace `Diffusion_framework/LFP/checkpoints/` with dataset's `Diffusion_framework/LFP/checkpoints/` (optional, for pre-trained models)
- Repeat the same for `NCM/type1/`, `NCM/type2/`, and `NCA/` folders

**Repository Structure:**
```
GDF-DIFF/
├── Diffusion_framework/
│   ├── LFP/
│   │   ├── data/                        # Train/validation/test datasets
│   │   ├── data_for_generated/          # Test set for reverse diffusion (generates structured .mat files)
│   │   ├── generated_data/              # Generated .mat outputs
│   │   ├── checkpoints/                 # Trained model weights
│   │   ├── output/                      # Visualization results and evaluation metrics
│   │   ├── GDF_data_processed.py        # Data preprocessing script
│   │   ├── GDF_train.py                 # Model training script
│   │   ├── GDF_test.py                  # Model testing and evaluation
│   │   ├── GDF_Reverse_diffusion.py     # Reverse diffusion sampling (.mat version of test script)
│   │   ├── GDF_diffusion_model.py       # Integrated diffusion model (combines forward/reverse processes)
│   │   ├── GDF_Unet.py                  # U-Net backbone implementation
│   │   ├── GDF_gaussian_diffusion.py    # Complete diffusion process (forward and reverse functions)
│   │   ├── GDF_visualization.py         # Result visualization tools
│   │   ├── GDF_main.py                  # Main execution script
│   │   └── README.md
│   ├── NCM/                         
│   │   ├── type1/                       # NCM battery type 1 (same structure as LFP)
│   │   │   ├── data/
│   │   │   ├── data_for_generated/
│   │   │   ├── generated_data/
│   │   │   ├── checkpoints/
│   │   │   ├── output/
│   │   │   ├── GDF_*.py                 # All GDF scripts
│   │   │   └── README.md
│   │   └── type2/                       # NCM battery type 2 (same structure as type1)
│   │       └── ...
│   └── NCA/                             # Same structure as LFP
│       └── ...
└── README.md
```

**For `SOH_estimation_framework/` folder:** The dataset on Zenodo has a **different structure** from this repository. To use the dataset:

**Extract and replace folders from Zenodo dataset one chemistry at a time:**

From Zenodo's complete structure:
```
dataset/SOH_estimation_framework/
├── LFP/
│   ├── data/
│   ├── data_generated/
│   ├── models/
│   └── original/
├── NCM/
│   ├── type1/
│   │   ├── data/
│   │   ├── data_generated/
│   │   ├── models/
│   │   └── original/
│   └── type2/
│       ├── data/
│       ├── data_generated/
│       ├── models/
│       └── original/
└── NCA/
    ├── data/
    ├── data_generated/
    ├── models/
    └── original/
```

**Replace into this repository (one at a time):**

For **LFP**:
- Replace `dataset/SOH_estimation_framework/LFP/data/` → `SOH_estimation_framework/data/`
- Replace `dataset/SOH_estimation_framework/LFP/data_generated/` → `SOH_estimation_framework/data_generated/`
- Replace `dataset/SOH_estimation_framework/LFP/models/` → `SOH_estimation_framework/models/`
- Replace `dataset/SOH_estimation_framework/LFP/original/` → `SOH_estimation_framework/original/`

For **NCM type1**:
- Replace `dataset/SOH_estimation_framework/NCM/type1/data/` → `SOH_estimation_framework/data/`
- Replace `dataset/SOH_estimation_framework/NCM/type1/data_generated/` → `SOH_estimation_framework/data_generated/`
- Replace `dataset/SOH_estimation_framework/NCM/type1/models/` → `SOH_estimation_framework/models/`
- Replace `dataset/SOH_estimation_framework/NCM/type1/original/` → `SOH_estimation_framework/original/`

For **NCM type2**:
- Replace `dataset/SOH_estimation_framework/NCM/type2/data/` → `SOH_estimation_framework/data/`
- Replace `dataset/SOH_estimation_framework/NCM/type2/data_generated/` → `SOH_estimation_framework/data_generated/`
- Replace `dataset/SOH_estimation_framework/NCM/type2/models/` → `SOH_estimation_framework/models/`
- Replace `dataset/SOH_estimation_framework/NCM/type2/original/` → `SOH_estimation_framework/original/`

For **NCA**:
- Replace `dataset/SOH_estimation_framework/NCA/data/` → `SOH_estimation_framework/data/`
- Replace `dataset/SOH_estimation_framework/NCA/data_generated/` → `SOH_estimation_framework/data_generated/`
- Replace `dataset/SOH_estimation_framework/NCA/models/` → `SOH_estimation_framework/models/`
- Replace `dataset/SOH_estimation_framework/NCA/original/` → `SOH_estimation_framework/original/`

**Repository Structure:**
```
GDF-DIFF/
└── SOH_estimation_framework/
    ├── data/                        # Battery cycling data (replace for each chemistry)
    ├── data_generated/              # Generated features for SOH estimation (replace for each chemistry)
    ├── models/                      # Trained model checkpoints (replace for each chemistry)
    ├── original/                    # Original/reference data (replace for each chemistry)
    ├── data_preprocess.py           # Data preprocessing script
    ├── model.py                     # SOH estimation model architecture
    ├── train_and_test.py            # Model training and testing
    ├── test_and_output.py           # Testing and result output
    └── README.md
```

**Note:** Run experiments separately for each battery chemistry by replacing the `data/`, `data_generated/`, `models/`, and `original/` folders before each run.
