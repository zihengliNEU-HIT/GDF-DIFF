import os
import argparse
import torch
import numpy as np

from GDF_data_processed import get_dataloaders
from GDF_gaussian_diffusion import GaussianDiffusion
from GDF_Unet import UNet
from GDF_diffusion_model import DiffusionModel


def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Battery CAF image diffusion model main program")
    
    # Basic parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Running mode")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=260, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=400, help="Diffusion timesteps")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    
    # Advanced parameters
    parser.add_argument("--scheduler", type=str, choices=["steplr", "cosine", "plateau", "none"], 
                        default="steplr", help="Learning rate scheduler type")
    parser.add_argument("--step_size", type=int, default=10, help="StepLR decay interval in epochs")
    parser.add_argument("--step_gamma", type=float, default=0.8, help="StepLR decay factor")
    parser.add_argument("--noise_schedule", type=str, choices=["linear", "cosine"], 
                        default="cosine", help="Noise schedule type")
    parser.add_argument("--clip_grad", type=float, default=1, help="Gradient clipping threshold")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--ema", type=float, default=0.99995, help="EMA coefficient")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--sample_idx", type=int, default=0, help="Visualization sample index")
    parser.add_argument("--cosine_s", type=float, default=0.008, help="Cosine schedule smoothing parameter")
    parser.add_argument("--cond_vector_dim", type=int, default=20, 
                        help="Conditioning vector dimension (0 to disable)")

    # DDIM sampling parameters
    parser.add_argument("--use_ddim", action="store_true", default=True, help="Use DDIM sampling")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM stochasticity parameter")

    return parser.parse_args()


def create_model(args):
    """
    Create diffusion model with U-Net architecture.
    
    Args:
        args: Training arguments
    Returns:
        Diffusion model instance
    """
    # Create U-Net network
    network = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=32,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=args.cond_vector_dim
    ).to(args.device)
    
    # Create EMA U-Net network
    ema_network = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=1,
        model_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        condition_emb_dim=32,
        use_attention=(False, False, True),
        num_res_blocks=2,
        cond_vector_dim=args.cond_vector_dim
    ).to(args.device)
    
    # Initialize model weights
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    network.apply(init_weights)
    ema_network.load_state_dict(network.state_dict())
    
    # Create Gaussian diffusion utility
    gdf_util = GaussianDiffusion(
        timesteps=args.timesteps,
        device=args.device,
        schedule_type=args.noise_schedule,
        cosine_s=args.cosine_s
    )
    
    # Create diffusion model
    diffusion_model = DiffusionModel(
        network=network,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=args.timesteps,
        ema=args.ema
    ).to(args.device)
    
    return diffusion_model


def main():
    """
    Main entry point for training and testing.
    """
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Execute based on running mode
    if args.mode == "train":
        from GDF_train import main as train_main
        train_main(args)
    
    elif args.mode == "test":
        from GDF_test import main as test_main
        test_main(args)


if __name__ == "__main__":
    main()