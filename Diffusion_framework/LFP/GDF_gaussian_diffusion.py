import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class GaussianDiffusion:
    """
    Gaussian Diffusion utility class for implementing forward and reverse diffusion processes.
    Supports both linear and cosine noise schedules, DDPM and DDIM sampling.
    
    Args:
        beta_start: Starting variance
        beta_end: Ending variance
        timesteps: Number of diffusion timesteps
        clip_min: Minimum value for clipping
        clip_max: Maximum value for clipping
        device: Device type (cuda or cpu)
        schedule_type: Noise schedule type, either "linear" or "cosine"
        cosine_s: Smoothing parameter for cosine schedule
    """
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.012,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        device="cuda",
        schedule_type="cosine",
        cosine_s: float = 0.008
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device
        self.schedule_type = schedule_type
        self.num_timesteps = int(timesteps)

        print(f"GaussianDiffusion initialized: schedule={schedule_type}, steps={timesteps}, beta=[{beta_start}, {beta_end}]")

        # Select noise schedule based on schedule_type
        if schedule_type == "linear":
            self.betas = np.linspace(
                beta_start,
                beta_end,
                timesteps,
                dtype=np.float64,
            )
            print("Using linear noise schedule")
        elif schedule_type == "cosine":
            # Cosine schedule for improved training stability
            steps = timesteps + 1
            s = float(cosine_s)
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(betas, 0.0001, 0.9999)
            print("Using cosine noise schedule for stable training")
        else:
            raise ValueError(f"Unsupported noise schedule type: {schedule_type}")

        # Compute diffusion coefficients
        alphas = 1.0 - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Convert to torch tensors
        self.betas = torch.tensor(self.betas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)

        # Coefficients for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32
        )
        self.log_one_minus_alphas_cumprod = torch.tensor(
            np.log(1.0 - alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_recip_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32
        )

        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas.numpy() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32)
        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32
        )
        self.posterior_mean_coef1 = torch.tensor(
            self.betas.numpy() * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=torch.float32,
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=torch.float32,
        )
        
        # Move all tensors to specified device
        self._to_device(device)
        print(f"All diffusion tensors moved to {device}")
        
        # Save noise schedule visualization
        try:
            os.makedirs("./output", exist_ok=True)
            self._save_noise_schedule("./output/noise_schedule.png")
        except Exception as e:
            print(f"Failed to save noise schedule plot: {e}")

    def _to_device(self, device):
        """
        Move all diffusion tensors to the specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
    
    def _save_noise_schedule(self, save_path):
        """
        Save visualization of noise schedule.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.betas.cpu().numpy())
        plt.title('Betas')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.alphas_cumprod.cpu().numpy())
        plt.title('Alphas Cumprod')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.sqrt_one_minus_alphas_cumprod.cpu().numpy())
        plt.title('Noise Ratio')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.posterior_variance.cpu().numpy())
        plt.title('Posterior Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Noise schedule plot saved to {save_path}")

    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at specified timesteps and reshape for broadcasting.
        
        Args:
            a: Coefficient tensor to extract from
            t: Timesteps to extract at
            x_shape: Shape of the current batch for broadcasting
        Returns:
            Extracted coefficients with proper shape for broadcasting
        """
        batch_size = x_shape[0]
        t = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        """
        Compute mean and variance for forward diffusion q(x_t | x_0).
        
        Args:
            x_start: Initial clean sample
            t: Current timestep
        Returns:
            Tuple of (mean, variance, log_variance)
        """
        x_start_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to clean samples.
        
        Args:
            x_start: Initial clean sample
            t: Current timestep
            noise: Gaussian noise to add, randomly generated if None
        Returns:
            Noisy sample at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_start_shape = x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noisy sample x_t and predicted noise.
        
        Args:
            x_t: Noisy sample at timestep t
            t: Current timestep
            noise: Predicted noise
        Returns:
            Predicted clean sample x_0
        """
        x_t_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Clean sample for posterior computation
            x_t: Noisy sample at timestep t
            t: Current timestep
        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        x_t_shape = x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t).
        
        Args:
            pred_noise: Predicted noise from the model
            x: Noisy sample at timestep t
            t: Current timestep
            clip_denoised: Whether to clip predicted x_0
        Returns:
            Tuple of (model_mean, posterior_variance, posterior_log_variance)
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        
        posterior_variance = posterior_variance.clamp(min=1e-20)
        posterior_log_variance = torch.log(posterior_variance)
        
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """
        Single step of reverse diffusion sampling.
        
        Args:
            pred_noise: Predicted noise from the model
            x: Current noisy sample
            t: Current timestep
            clip_denoised: Whether to clip denoised output
        Returns:
            Sample at timestep t-1
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        
        variance = (0.5 * model_log_variance).exp().clamp(max=0.99)
        return model_mean + nonzero_mask * variance * noise
        
    @torch.no_grad()
    def p_sample_loop(self, shape, model, condition=None, clip_denoised=True):
        """
        Complete DDPM sampling loop from pure noise to clean sample.
        
        Args:
            shape: Shape of samples to generate
            model: Trained diffusion model
            condition: Optional conditional input
            clip_denoised: Whether to clip denoised samples
        Returns:
            Generated samples
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.num_timesteps)):
            timesteps = torch.full((b,), i, device=device, dtype=torch.long)
            pred_noise = model(img, timesteps, condition)
            img = self.p_sample(pred_noise, img, timesteps, clip_denoised=clip_denoised)
        
        return img
    
    def noise_estimation_loss(self, model, x_start, t, condition=None, noise=None):
        """
        Compute training loss combining noise prediction and reconstruction objectives.
        Uses adaptive weighting and timestep-based weighting for improved PSNR.
        
        Args:
            model: Diffusion model
            x_start: Clean target samples
            t: Timesteps
            condition: Optional conditional input
            noise: Noise samples, randomly generated if None
        Returns:
            Combined training loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)
                
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t, condition)
        
        # Noise prediction loss
        mse_loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        l1_loss = torch.nn.functional.l1_loss(noise, predicted_noise)
        noise_loss = 0.9 * mse_loss + 0.1 * l1_loss
        
        # Reconstruction loss for direct image quality optimization
        predicted_x0 = self.predict_start_from_noise(x_noisy, t, predicted_noise)
        predicted_x0 = torch.clamp(predicted_x0, self.clip_min, self.clip_max)
        
        recon_mse = torch.nn.functional.mse_loss(predicted_x0, x_start)
        recon_l1 = torch.nn.functional.l1_loss(predicted_x0, x_start)
        recon_loss = 0.8 * recon_mse + 0.2 * recon_l1
        
        # Adaptive weight scheduling
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        self._step_counter += 1
        # 130000: warmup steps for loss weight transition
        progress = min(self._step_counter / 130000, 1.0) 
        noise_weight = 0.7 - 0.3 * progress
        recon_weight = 0.3 + 0.3 * progress
        
        # Timestep-based weighting
        t_normalized = t.float() / self.timesteps
        time_weight = torch.exp(-2 * t_normalized)
        time_weight = time_weight.view(-1, 1, 1, 1)
        
        weighted_recon_loss = (recon_loss * time_weight).mean()
        
        # Combined loss
        loss = (
            noise_weight * noise_loss +
            recon_weight * weighted_recon_loss
        )
        
        # Monitoring every 2000 steps
        if self._step_counter % 2000 == 0:
            with torch.no_grad():
                pred_psnr = (predicted_x0 + 1.0) / 2.0
                target_psnr = (x_start + 1.0) / 2.0
                
                mse_psnr = torch.mean((pred_psnr - target_psnr) ** 2)
                if mse_psnr > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_psnr))
                else:
                    psnr = torch.tensor(100.0)
                
                print(f"\nLoss Monitor Step {self._step_counter}")
                print(f"Noise loss: {noise_loss.item():.6f}")
                print(f"Recon loss: {weighted_recon_loss.item():.6f}")
                print(f"Weights noise: {noise_weight:.3f}, recon: {recon_weight:.3f}")
                print(f"Total loss: {loss.item():.6f}")
                print(f"Current PSNR: {psnr.item():.2f} dB")
                
                if psnr.item() > 38:
                    print("Target achieved: PSNR > 38dB")
                elif psnr.item() > 35:
                    print("Good progress: PSNR > 35dB")
                elif psnr.item() > 30:
                    print("Steady improvement: PSNR > 30dB")
        
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        cond=None,
        steps: int = 50,
        eta: float = 0.0,
        x_T: torch.Tensor = None,
        clip_denoised: bool = True,
    ):
        """
        DDIM sampling for fast generation with fewer steps.
        
        Args:
            model: Trained diffusion model
            cond: Conditional input
            steps: Number of sampling steps (typically 20-100)
            eta: Stochasticity parameter (0 = deterministic DDIM)
            x_T: Initial Gaussian noise, randomly generated if None
            clip_denoised: Whether to clip output to valid range
        Returns:
            Generated clean sample x_0
        """
        device = self.betas.device
        
        # Generate subsequence of timesteps
        step_size = self.num_timesteps // steps
        idx = torch.arange(self.num_timesteps - 1, -1, -step_size, device=device).long()
        
        if idx[-1] != 0:
            idx[-1] = 0
        
        # Initialize with noise
        if x_T is None:
            if cond is not None:
                batch_size = cond.shape[0]
                h, w = cond.shape[-2:]
                x = torch.randn((batch_size, 1, h, w), device=device)
            else:
                raise ValueError("Either x_T or cond must be provided to infer shape.")
        else:
            x = x_T.to(device)
        
        # DDIM sampling loop
        for i, t in enumerate(idx):
            t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps_theta = model(x, t_tensor, cond)
            
            alpha_bar_t = self._extract(self.alphas_cumprod, t_tensor, x.shape)
            alpha_bar_t_sqrt = self._extract(self.sqrt_alphas_cumprod, t_tensor, x.shape)
            one_minus_alpha_bar_t_sqrt = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
            
            x0_hat = (x - one_minus_alpha_bar_t_sqrt * eps_theta) / alpha_bar_t_sqrt
            
            if clip_denoised:
                x0_hat = torch.clamp(x0_hat, self.clip_min, self.clip_max)
            
            if i == len(idx) - 1:
                return x0_hat
            
            t_prev = idx[i + 1]
            t_prev_tensor = torch.full((x.size(0),), t_prev, device=device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev_tensor, x.shape)
            
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * 
                                    (1 - alpha_bar_t / alpha_bar_prev))
            
            noise = torch.randn_like(x) if eta > 0 else 0
            
            x = (torch.sqrt(alpha_bar_prev) * x0_hat +
                    torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_theta +
                    sigma * noise)
        
        return x