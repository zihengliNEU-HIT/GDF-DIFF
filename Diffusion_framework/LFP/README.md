## Usage

### Training

To train the diffusion model (using LFP as an example), run:
```bash
cd Diffusion_framework/LFP
python GDF_main.py --mode train --num_epochs 260 --batch_size 16 --lr 1e-3
```

**Key training parameters:**
- `--mode train`: Set to training mode
- `--num_epochs`: Number of training epochs (default: 260)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--timesteps`: Diffusion timesteps (default: 400)
- `--checkpoint_dir`: Directory to save model checkpoints (default: ./checkpoints)

**Example with custom parameters:**
```bash
cd Diffusion_framework/LFP
python GDF_main.py \
  --mode train \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints \
  --num_epochs 300 \
  --batch_size 32 \
  --lr 5e-4 \
  --timesteps 500
```

### Testing

To test the trained model (using LFP as an example), run:
```bash
cd Diffusion_framework/LFP
python GDF_main.py --mode test --checkpoint_path ./checkpoints/model_epoch_60.pt
```

**Key testing parameters:**
- `--mode test`: Set to testing mode
- `--checkpoint_path`: Path to the trained model checkpoint (e.g., `./checkpoints/model_epoch_60.pt`)
- `--output_dir`: Directory to save test results (default: ./output)

**Example:**
```bash
cd Diffusion_framework/LFP
python GDF_main.py \
  --mode test \
  --checkpoint_path ./checkpoints/model_epoch_60.pt \
  --output_dir ./output \
  --batch_size 16
```

**Note:** If `--checkpoint_path` is not specified, the script will automatically look for `model_epoch_60.pt` in the checkpoint directory.

**For other battery chemistries:** Navigate to the corresponding directory (e.g., `Diffusion_framework/NCM/type1` or `Diffusion_framework/NCA`) and run the same commands.

### Additional Parameters

**DDIM Sampling (for faster generation):**
```bash
python GDF_main.py --mode test --use_ddim --ddim_steps 50 --ddim_eta 0.0
```

**Learning Rate Scheduler:**
```bash
python GDF_main.py --mode train --scheduler steplr --step_size 10 --step_gamma 0.8
```

**Noise Schedule:**
```bash
python GDF_main.py --mode train --noise_schedule cosine --cosine_s 0.008
```

For a complete list of parameters, run:
```bash
python GDF_main.py --help
```
