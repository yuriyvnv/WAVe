"""
Attentive pooling mechanism for WAVe model.

This module implements learned attention-based pooling that computes a weighted
average of sequence representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePooling(nn.Module):
    """
    Attentive pooling for variable-length sequences.

    Instead of simple mean/max pooling, this module learns attention weights
    to compute a weighted average, allowing the model to focus on important
    parts of the sequence.

    Args:
        hidden_size (int): Dimension of input hidden states

    Example:
        >>> pooling = AttentivePooling(hidden_size=768)
        >>> hidden_states = torch.randn(4, 50, 768)  # (batch, seq_len, hidden)
        >>> mask = torch.ones(4, 50)  # (batch, seq_len)
        >>> pooled = pooling(hidden_states, mask)  # Shape: (4, 768)
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        # Two-layer MLP for computing attention scores
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)  # Reduce to scalar score per token
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Apply attentive pooling to sequence.

        Args:
            hidden_states: Sequence tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch_size, seq_len)
                           where 1 = valid token, 0 = padding

        Returns:
            Pooled representation of shape (batch_size, hidden_size)
        """
        # Compute attention scores for each token
        # (batch, seq_len, hidden) -> (batch, seq_len, 1) -> (batch, seq_len)
        attention_scores = self.attention(hidden_states).squeeze(-1)

        # Apply mask to prevent attending to padding
        if attention_mask is not None:
            # Set padding positions to -inf so softmax gives them 0 weight
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Normalize scores to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Weighted sum: (batch, 1, seq_len) @ (batch, seq_len, hidden) -> (batch, 1, hidden)
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states                     # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)

        return pooled_output
