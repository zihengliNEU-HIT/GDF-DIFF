import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Weight initialization function
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for encoding timestep information
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: [batch_size] timestep tensor
        Returns:
            positional embeddings [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock(nn.Module):
    """
    Self-attention block for enhancing feature extraction
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # Debug: FiLM print switch and frequency (optional)
        self.debug_film = True
        self._film_dbg_every = 500
        self._film_dbg_step = 0

        # Initialize weights
        self.query.apply(init_weights)
        self.key.apply(init_weights)
        self.value.apply(init_weights)
        self.proj_out.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, height, width]
        Returns:
            enhanced features [batch_size, channels, height, width]
        """
        x_norm = self.norm(x)

        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)

        batch, c, h, w = q.shape
        q = q.reshape(batch, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        k = k.reshape(batch, c, -1)                   # [B, C, H*W]
        v = v.reshape(batch, c, -1).permute(0, 2, 1)  # [B, H*W, C]

        scale = 1.0 / math.sqrt(c)
        attention = torch.bmm(q, k) * scale           # [B, H*W, H*W]
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(attention, v)                 # [B, H*W, C]
        out = out.permute(0, 2, 1).reshape(batch, c, h, w)  # [B, C, H, W]

        out = self.proj_out(out)

        return x + out

class DownBlock(nn.Module):
    """
    Downsampling block used in the encoder part of U-Net
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, cond_vector_dim=0):
        super().__init__()
        self.use_attention = use_attention

        # Debug: FiLM statistics (optional)
        self.debug_film = True
        self._film_dbg_every = 1000
        self._film_dbg_step = 0

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Convolutional blocks
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Attention module
        if use_attention:
            self.attention = AttentionBlock(out_channels)

        # FiLM mapping for scalar condition vector (gamma, beta)
        self.has_cond = cond_vector_dim > 0
        if self.has_cond:
            self.cond_mlp1 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, out_channels * 2))
            self.cond_mlp2 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, out_channels * 2))
            nn.init.zeros_(self.cond_mlp1[-1].weight); nn.init.zeros_(self.cond_mlp1[-1].bias)
            nn.init.zeros_(self.cond_mlp2[-1].weight); nn.init.zeros_(self.cond_mlp2[-1].bias)

        # Initialize weights
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        if isinstance(self.residual_conv, nn.Conv2d):
            self.residual_conv.apply(init_weights)

    def _apply_film(self, h, cond_vec, mlp):
        if (cond_vec is None) or (not self.has_cond):
            return h

        # Prevent dtype/device mismatch
        cond_vec = cond_vec.to(h.dtype).to(h.device)

        gb = mlp(cond_vec)                          # [B, 2C]
        gamma, beta = gb.chunk(2, dim=1)            # [B, C], [B, C]
        gamma = gamma[:, :, None, None]
        beta  = beta [:, :, None, None]

        # Debug: throttled printing for gamma/beta statistics
        if getattr(self, "debug_film", False):
            self._film_dbg_step += 1
            if (self._film_dbg_step % getattr(self, "_film_dbg_every", 1000)) == 0:
                with torch.no_grad():
                    print(
                        f"[DownBlock FiLM] calls={self._film_dbg_step} "
                        f"gamma mean={gamma.mean():+.4f}, std={gamma.std():.4f} | "
                        f"beta mean={beta.mean():+.4f}, std={beta.std():.4f}"
                    )

        scale = 1.2
        return h * (1.0 + scale * gamma) + scale * beta

    def forward(self, x, time_emb, cond_vec=None):
        """
        Args:
            x: [batch_size, in_channels, height, width]
            time_emb: [batch_size, time_emb_dim]
        Returns:
            downsampled features [batch_size, out_channels, height, width]
        """
        residual = self.residual_conv(x)

        # ---- Layer 1 ----
        h = self.conv1(x)
        h = self.norm1(h)

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp1") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp1)

        h = F.silu(h)

        # Add time embedding projection
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj

        # ---- Layer 2 ----
        h = self.conv2(h)
        h = self.norm2(h)

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp2") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp2)

        h = F.silu(h)

        # Attention layer
        if self.use_attention:
            h = self.attention(h)

        return h + residual

class UpBlock(nn.Module):
    """
    Upsampling block used in the decoder part of U-Net
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False, cond_vector_dim=0):
        super().__init__()
        self.use_attention = use_attention

        # Debug: FiLM statistics (optional)
        self.debug_film = True
        self._film_dbg_every = 1000
        self._film_dbg_step = 0

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Upsampling convolutional blocks
        self.conv1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Attention module
        if use_attention:
            self.attention = AttentionBlock(out_channels)

        # FiLM mapping for cond_vec
        self.has_cond = cond_vector_dim > 0
        if self.has_cond:
            self.cond_mlp1 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, out_channels * 2))
            self.cond_mlp2 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, out_channels * 2))
            nn.init.zeros_(self.cond_mlp1[-1].weight); nn.init.zeros_(self.cond_mlp1[-1].bias)
            nn.init.zeros_(self.cond_mlp2[-1].weight); nn.init.zeros_(self.cond_mlp2[-1].bias)

        # Initialize weights
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        if isinstance(self.residual_conv, nn.Conv2d):
            self.residual_conv.apply(init_weights)

    def _apply_film(self, h, cond_vec, mlp):
        if (cond_vec is None) or (not self.has_cond):
            return h

        # Prevent dtype/device mismatch
        cond_vec = cond_vec.to(h.dtype).to(h.device)

        gb = mlp(cond_vec)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta [:, :, None, None]

        # Debug: throttled printing
        if getattr(self, "debug_film", False):
            self._film_dbg_step += 1
            if (self._film_dbg_step % getattr(self, "_film_dbg_every", 1000)) == 0:
                with torch.no_grad():
                    print(
                        f"[UpBlock FiLM] calls={self._film_dbg_step} "
                        f"gamma mean={gamma.mean():+.4f}, std={gamma.std():.4f} | "
                        f"beta mean={beta.mean():+.4f}, std={beta.std():.4f}"
                    )
        scale = 1.2
        return h * (1.0 + scale * gamma) + scale * beta

    def forward(self, x, skip_x, time_emb, cond_vec=None):
        """
        Args:
            x: [batch_size, in_channels, height, width]
            skip_x: skip connection from encoder [batch_size, out_channels, height, width]
            time_emb: [batch_size, time_emb_dim]
        Returns:
            upsampled features [batch_size, out_channels, height, width]
        """
        residual = self.residual_conv(x)

        h = torch.cat([x, skip_x], dim=1)

        h = self.norm1(self.conv1(h))

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp1") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp1)

        h = F.silu(h)

        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj

        h = self.norm2(self.conv2(h))

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp2") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp2)

        h = F.silu(h)

        if self.use_attention:
            h = self.attention(h)

        return h + residual

class MiddleBlock(nn.Module):
    """
    Middle block connecting the encoder and decoder of U-Net
    """
    def __init__(self, channels, time_emb_dim, cond_vector_dim=0):
        super().__init__()

        # Debug: FiLM statistics (optional)
        self.debug_film = True
        self._film_dbg_every = 1000
        self._film_dbg_step = 0

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )

        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)

        # Attention module
        self.attention = AttentionBlock(channels)

        # Second convolution
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

        # FiLM mapping for cond_vec
        self.has_cond = cond_vector_dim > 0
        if self.has_cond:
            self.cond_mlp1 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, channels * 2))
            self.cond_mlp2 = nn.Sequential(nn.SiLU(), nn.Linear(cond_vector_dim, channels * 2))
            nn.init.zeros_(self.cond_mlp1[-1].weight); nn.init.zeros_(self.cond_mlp1[-1].bias)
            nn.init.zeros_(self.cond_mlp2[-1].weight); nn.init.zeros_(self.cond_mlp2[-1].bias)

        # Initialize weights
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)

    def _apply_film(self, h, cond_vec, mlp):
        if (cond_vec is None) or (not self.has_cond):
            return h

        # Prevent dtype/device mismatch
        cond_vec = cond_vec.to(h.dtype).to(h.device)

        gb = mlp(cond_vec)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta [:, :, None, None]

        # Debug: throttled printing
        if getattr(self, "debug_film", False):
            self._film_dbg_step += 1
            if (self._film_dbg_step % getattr(self, "_film_dbg_every", 1000)) == 0:
                with torch.no_grad():
                    print(
                        f"[MiddleBlock FiLM] calls={self._film_dbg_step} "
                        f"gamma mean={gamma.mean():+.4f}, std={gamma.std():.4f} | "
                        f"beta mean={beta.mean():+.4f}, std={beta.std():.4f}"
                    )

        scale = 1.2
        return h * (1.0 + scale * gamma) + scale * beta

    def forward(self, x, time_emb, cond_vec=None):
        """
        Args:
            x: [batch_size, channels, height, width]
            time_emb: [batch_size, time_emb_dim]
        Returns:
            processed features [batch_size, channels, height, width]
        """
        residual = x

        h = self.norm1(self.conv1(x))

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp1") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp1)

        h = F.silu(h)

        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj

        h = self.attention(h)

        h = self.norm2(self.conv2(h))

        # Insert FiLM after GroupNorm (optional)
        if hasattr(self, "cond_mlp2") and getattr(self, "has_cond", False):
            h = self._apply_film(h, cond_vec, self.cond_mlp2)

        h = F.silu(h)

        return h + residual

class Upsample(nn.Module):
    """
    Upsampling module
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, height, width]
        Returns:
            upsampled features [batch_size, channels, height*2, width*2]
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class Downsample(nn.Module):
    """
    Downsampling module
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2)
        self.conv.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch_size, channels, height, width]
        Returns:
            downsampled features [batch_size, channels, height/2, width/2]
        """
        return self.conv(x)

class ConditionEmbedding(nn.Module):
    """
    Condition embedding module for processing conditional inputs (5-channel image)
    """
    def __init__(self, condition_channels, emb_dim):
        super().__init__()
        self.conv_in = nn.Conv2d(condition_channels, emb_dim, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)

        # Initialize weights
        self.conv_in.apply(init_weights)
        self.conv_mid.apply(init_weights)
        self.conv_out.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch_size, condition_channels, height, width]
        Returns:
            condition embeddings [batch_size, emb_dim, height, width]
        """
        h = F.silu(self.conv_in(x))
        h = F.silu(self.conv_mid(h))
        h = self.conv_out(h)
        return h

class UNet(nn.Module):
    """
    Conditional U-Net model for diffusion
    """
    def __init__(
        self,
        in_channels=1,           # number of input channels (noisy image)
        out_channels=1,          # number of output channels (predicted noise)
        condition_channels=5,    # number of conditional input channels
        model_channels=64,       # base channel width
        channel_mults=(1, 2, 4, 8),  # channel multipliers at each resolution
        time_emb_dim=256,        # time embedding dimension
        condition_emb_dim=64,    # condition embedding dimension
        use_attention=(False, True, True, True),  # whether to use attention at each stage
        num_res_blocks=2,        # number of residual blocks per stage
        cond_vector_dim=0,       # dimension of extra condition vector (0 disables FiLM)
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.condition_emb_dim = condition_emb_dim
        self.num_res_blocks = num_res_blocks
        self.cond_vector_dim = cond_vector_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Condition embedding
        self.condition_embedding = ConditionEmbedding(condition_channels, condition_emb_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Project condition embedding channels to match backbone channels
        self.cond_proj = nn.Conv2d(condition_emb_dim, model_channels, kernel_size=1)
        self.cond_proj.apply(init_weights)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = model_channels

        for i, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(in_ch, out_ch, time_emb_dim, use_attention[i], cond_vector_dim=self.cond_vector_dim)
                )
                in_ch = out_ch
            if i != len(channel_mults) - 1:
                self.downsamplers.append(Downsample(in_ch))

        # Middle block
        self.middle_block = MiddleBlock(in_ch, time_emb_dim, cond_vector_dim=self.cond_vector_dim)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    UpBlock(in_ch, out_ch, time_emb_dim, use_attention[i], cond_vector_dim=self.cond_vector_dim)
                )
                in_ch = out_ch
            if i != 0:
                self.upsamplers.append(Upsample(in_ch))
                in_ch = in_ch

        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self.conv_in.apply(init_weights)
        self.conv_out.apply(init_weights)

    def forward(self, x, timesteps, condition=None, cond_vec=None):
        """
        Args:
            x: [batch_size, in_channels, height, width] noisy image
            timesteps: [batch_size] diffusion timesteps
            condition: [batch_size, condition_channels, height, width] conditional image
            cond_vec: [batch_size, D] optional condition vector (D=cond_vector_dim)
        Returns:
            predicted noise [batch_size, out_channels, height, width]
        """
        time_emb = self.time_embedding(timesteps)

        cond_emb = None
        if condition is not None:
            cond_emb = self.condition_embedding(condition)

        h = self.conv_in(x)

        if cond_emb is not None:
            h = h + self.cond_proj(cond_emb)

        skips = [h]

        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h, time_emb, cond_vec=cond_vec)
            skips.append(h)
            if (i + 1) % self.num_res_blocks == 0 and (i // self.num_res_blocks) < len(self.downsamplers):
                h = self.downsamplers[i // self.num_res_blocks](h)

        h = self.middle_block(h, time_emb, cond_vec=cond_vec)

        for i, up_block in enumerate(self.up_blocks):
            if i > 0 and i % self.num_res_blocks == 0:
                h = self.upsamplers[i // self.num_res_blocks - 1](h)

            skip_h = skips.pop()
            h = up_block(h, skip_h, time_emb, cond_vec=cond_vec)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h

# Test code (optional)
if __name__ == "__main__":
    # Create model example (FiLM enabled with cond_vector_dim=8)
    model = UNet(
        in_channels=1,
        out_channels=1,
        condition_channels=5,
        model_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
        condition_emb_dim=64,
        use_attention=(False, True, True, True),
        num_res_blocks=2,
        cond_vector_dim=8,
    )

    B, H, W = 2, 128, 128
    x = torch.randn(B, 1, H, W)
    t = torch.tensor([500, 250])
    condition = torch.randn(B, 5, H, W)
    cond_vec = torch.randn(B, 8)

    out = model(x, t, condition, cond_vec=cond_vec)
    print("out:", out.shape)
