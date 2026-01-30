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
│   │   ├── data/                    # Train/validation/test datasets
│   │   ├── data_for_generated/      # Test set for reverse diffusion (generates structured .mat files)
│   │   ├── generated_data/          # Generated .mat outputs
│   │   ├── checkpoints/             # Trained model weights
│   │   └── output/                  # Visualization results and evaluation metrics
│   ├── NCM/                         
│   │   ├── type1/                   # NCM battery type 1
│   │   │   ├── data/
│   │   │   ├── data_for_generated/
│   │   │   ├── generated_data/
│   │   │   ├── checkpoints/
│   │   │   └── output/
│   │   └── type2/                   # NCM battery type 2
│   │       ├── data/
│   │       ├── data_for_generated/
│   │       ├── generated_data/
│   │       ├── checkpoints/
│   │       └── output/
│   └── NCA/                         # Same structure as LFP
│       ├── data/
│       ├── data_for_generated/
│       ├── generated_data/
│       ├── checkpoints/
│       └── output/
├── GDF_data_processed.py            # Data preprocessing script
├── GDF_train.py                     # Model training script
├── GDF_test.py                      # Model testing and evaluation
├── GDF_Reverse_diffusion.py         # Reverse diffusion sampling (.mat version of test script)
├── GDF_diffusion_model.py           # Integrated diffusion model (combines forward/reverse processes)
├── GDF_Unet.py                      # U-Net backbone implementation
├── GDF_gaussian_diffusion.py        # Complete diffusion process (forward and reverse functions)
├── GDF_visualization.py             # Result visualization tools
└── README.md
```
