"""
Enhanced projection layers for WAVe model.

This module contains the projection layer implementation that maps encoder outputs
to a shared embedding space.
"""

import torch.nn as nn


class EnhancedProjection(nn.Module):
    """
    Enhanced projection layer with two-layer MLP architecture.

    Following SimCLR's design (Chen et al., 2020), this projection uses an
    expansion-compression architecture with intermediate non-linearity to
    improve representation quality for contrastive learning.

    Args:
        input_dim (int): Input dimension from encoder
        projection_dim (int): Target projection dimension (shared embedding space)
        hidden_dim (int, optional): Hidden dimension. If None, defaults to projection_dim * 2
        dropout (float): Dropout probability
        activation (str): Activation function - 'gelu' or 'relu'

    Example:
        >>> projection = EnhancedProjection(input_dim=768, projection_dim=512)
        >>> x = torch.randn(32, 768)  # batch_size=32, hidden_size=768
        >>> out = projection(x)  # Shape: (32, 512)
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        # Expansion: hidden_dim is 2x projection_dim by default (SimCLR design)
        if hidden_dim is None:
            hidden_dim = projection_dim * 2

        # Activation function
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'gelu' or 'relu'.")

        # Two-layer MLP with expansion-compression
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # Expand
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),  # Compress
            nn.LayerNorm(projection_dim)            # Normalize
        )

    def forward(self, x):
        """
        Project input to shared embedding space.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Projected tensor of shape (batch_size, projection_dim)
        """
        return self.projection(x)
