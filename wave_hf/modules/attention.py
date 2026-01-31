"""
Cross-modal attention mechanisms for WAVe model.

This module implements cross-modal attention that allows one modality to attend
to another (e.g., text attending to audio features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal multi-head attention mechanism.

    Enables one modality (query) to attend to another modality (key/value),
    allowing information flow between text and audio representations.

    Args:
        dim (int): Dimension of input/output features
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability

    Example:
        >>> attention = CrossModalAttention(dim=768, num_heads=8)
        >>> text = torch.randn(4, 20, 768)   # (batch, text_len, dim)
        >>> audio = torch.randn(4, 100, 768) # (batch, audio_len, dim)
        >>> attended = attention(text, audio)  # Text attending to audio
        >>> # Output shape: (4, 20, 768) - same as query
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        # Multi-head attention projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Xavier uniform initialization for stability
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, context, attention_mask=None):
        """
        Apply cross-modal attention from x (query) to context (key/value).

        Args:
            x: Query tensor of shape (batch_size, seq_len_q, dim)
            context: Key/Value tensor of shape (batch_size, seq_len_kv, dim)
            attention_mask: Optional mask of shape (batch_size, seq_len_kv)
                           where 1 = attend, 0 = mask out

        Returns:
            Output tensor of shape (batch_size, seq_len_q, dim)
        """
        batch_size = x.shape[0]

        # Project and reshape for multi-head attention
        # (batch, seq, dim) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_kv)
        # -> (batch, heads, seq_q, seq_kv)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: (batch, 1, 1, seq_kv)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert 0s to -inf for softmax masking
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, heads, seq_q, seq_kv) @ (batch, heads, seq_kv, head_dim)
        # -> (batch, heads, seq_q, head_dim)
        output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq_q, head_dim) -> (batch, seq_q, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final projection
        output = self.out_proj(output)

        return output
