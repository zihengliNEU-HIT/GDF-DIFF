import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransformer(nn.Module):
    """
    Hybrid CNN–Transformer architecture for battery capacity estimation.

    Args:
        img_size (tuple): Input image resolution (H, W)
        hidden_dim (int): Latent feature dimension
        num_layers (int): Number of Transformer encoder layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, img_size=(128, 128), hidden_dim=128,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()

        # === Convolutional feature extractor ===
        self.cnn = nn.Sequential(
            # Convolution block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolution block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolution block 3
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Feature map resolution after two pooling operations
        self.feature_size = img_size[0] // 4

        # Token sequence length 
        self.seq_len = self.feature_size * self.feature_size

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # === Regression head ===
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (Tensor): Input GAF images of shape [B, 1, H, W]

        Returns:
            Tensor: Normalized capacity prediction of shape [B, 1]
        """
        # Extract features
        features = self.cnn(x)

        # Reshape feature map 
        batch_size = features.size(0)
        features = features.view(batch_size, -1, self.seq_len).permute(0, 2, 1)

        # Add positional encoding
        features = features + self.pos_embedding
        features = self.dropout(features)

        # Global dependency modeling 
        features = self.transformer_encoder(features)

        # Global average pooling 
        features = torch.mean(features, dim=1)
        features = self.norm(features)

        # Linear regression output
        output = self.fc(features)

        return output


def count_parameters(model):
    """
    Count the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # check 
    batch_size = 4
    img_size = 128
    x = torch.randn(batch_size, 1, img_size, img_size)

    model = CNNTransformer(img_size=(img_size, img_size))
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
