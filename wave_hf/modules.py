"""
WAVe model modules.

This module contains the core architectural components of the WAVe model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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

"""
Word-level alignment module for WAVe model.

This is the CORE INNOVATION of WAVe - aligning text words with audio frames
using multi-head attention and scoring alignment quality with multi-head GLU.
"""

import torch
import torch.nn as nn


class WordLevelAlignmentModule(nn.Module):
    """
    Word-level alignment module with multi-head attention and GLU scoring.

    This module is the key innovation of WAVe. It:
    1. Projects text and audio to a shared alignment space
    2. Uses multi-head attention to align each word with relevant audio frames
    3. Enriches word representations with aligned audio context
    4. Scores alignment quality using multi-head Gated Linear Units (GLU)

    The alignment scores indicate how well each word corresponds to the audio,
    enabling detection of synthesis errors like mispronunciations or omissions.

    Args:
        text_hidden_dim (int): Hidden dimension of text encoder (e.g., 768 for RoBERTa)
        audio_hidden_dim (int): Hidden dimension of audio encoder (e.g., 1024 for Wav2Vec2-BERT)
        alignment_dim (int): Dimension of alignment space (typically same as projection_dim)
        num_heads (int): Number of attention heads for alignment (default: 6)
        dropout (float): Dropout probability
        n_glu_heads (int): Number of GLU heads for scoring (default: 4)

    Example:
        >>> alignment = WordLevelAlignmentModule(
        ...     text_hidden_dim=768,
        ...     audio_hidden_dim=1024,
        ...     alignment_dim=768,
        ...     num_heads=6,
        ...     n_glu_heads=4
        ... )
        >>> text_hidden = torch.randn(4, 20, 768)    # (batch, text_len, 768)
        >>> audio_hidden = torch.randn(4, 100, 1024) # (batch, audio_len, 1024)
        >>> text_mask = torch.ones(4, 20)
        >>> audio_mask = torch.ones(4, 100)
        >>>
        >>> aligned_repr, align_scores, align_matrix = alignment(
        ...     text_hidden, audio_hidden, text_mask, audio_mask
        ... )
        >>> # aligned_repr: (4, 20, 768) - enriched word representations
        >>> # align_scores: (4, 20) - per-word quality scores
        >>> # align_matrix: (4, 20, 100) - full attention matrix
    """

    def __init__(
        self,
        text_hidden_dim: int,
        audio_hidden_dim: int,
        alignment_dim: int,
        num_heads: int = 6,
        dropout: float = 0.1,
        n_glu_heads: int = 4
    ):
        super().__init__()

        self.text_hidden_dim = text_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.alignment_dim = alignment_dim
        self.num_heads = num_heads
        self.n_glu_heads = n_glu_heads

        # Project text and audio to common alignment space
        self.text_projection = nn.Linear(text_hidden_dim, alignment_dim)
        self.audio_projection = nn.Linear(audio_hidden_dim, alignment_dim)

        # Multi-head attention for word-to-audio alignment
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=alignment_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important: batch dimension comes first
        )

        # Output projection and normalization
        self.output_projection = nn.Linear(alignment_dim, alignment_dim)
        self.layer_norm = nn.LayerNorm(alignment_dim)

        # ===== MULTI-HEAD GLU SCORER =====
        # Learned temperature for sigmoid gating
        self.log_tau = nn.Parameter(torch.zeros(()))

        # GLU value and gate projections
        # Input: concatenated [original_query, aligned_representation]
        # Output: n_glu_heads separate projections
        out_dim = alignment_dim * n_glu_heads
        self.val = nn.Linear(alignment_dim * 2, out_dim)   # Value stream
        self.gate = nn.Linear(alignment_dim * 2, out_dim)  # Gate stream

        # Final 1-logit scorer (reduces multi-head output to single score)
        self.proj = nn.Linear(out_dim, 1)

    def forward(
        self,
        text_hidden_states,
        audio_hidden_states,
        text_attention_mask=None,
        audio_attention_mask=None
    ):
        """
        Compute word-level alignment between text and audio.

        Args:
            text_hidden_states: Text representations of shape (batch, text_len, text_hidden_dim)
            audio_hidden_states: Audio representations of shape (batch, audio_len, audio_hidden_dim)
            text_attention_mask: Text mask of shape (batch, text_len) where 1=valid, 0=padding
            audio_attention_mask: Audio mask of shape (batch, audio_len) where 1=valid, 0=padding

        Returns:
            tuple containing:
                - aligned_representations: Text enriched with audio context (batch, text_len, alignment_dim)
                - alignment_scores: Per-word quality scores (batch, text_len)
                - alignment_matrix: Full attention matrix (batch, text_len, audio_len)
        """
        batch_size, text_len, _ = text_hidden_states.shape
        _, audio_len, _ = audio_hidden_states.shape

        # Project to common alignment space
        text_proj = self.text_projection(text_hidden_states)    # (batch, text_len, alignment_dim)
        audio_proj = self.audio_projection(audio_hidden_states) # (batch, audio_len, alignment_dim)

        # Convert masks to format needed by PyTorch MultiheadAttention
        # (MultiheadAttention uses key_padding_mask where True = ignore)
        if text_attention_mask is not None:
            text_key_padding_mask = (1.0 - text_attention_mask).bool()
        else:
            text_key_padding_mask = None

        if audio_attention_mask is not None:
            audio_key_padding_mask = (1.0 - audio_attention_mask).bool()
        else:
            audio_key_padding_mask = None

        # ===== WORD-TO-AUDIO ATTENTION =====
        # Text tokens (queries) attend to audio frames (keys/values)
        aligned_representations, alignment_weights = self.alignment_attention(
            query=text_proj,                        # What we're aligning (words)
            key=audio_proj,                         # What we're aligning to (audio frames)
            value=audio_proj,                       # Information to aggregate
            key_padding_mask=audio_key_padding_mask,  # Mask out audio padding
            need_weights=True,
            average_attn_weights=False              # Return all attention heads
        )

        # Average attention weights across heads to get final alignment matrix
        # (batch, num_heads, text_len, audio_len) -> (batch, text_len, audio_len)
        alignment_matrix = alignment_weights.mean(dim=1)

        # Apply residual connection and layer normalization
        aligned_representations = self.layer_norm(
            text_hidden_states + self.output_projection(aligned_representations)
        )

        # ===== MULTI-HEAD GLU CONFIDENCE SCORER =====
        # Concatenate original query with aligned representation
        x_in = torch.cat([text_proj, aligned_representations], dim=-1)  # (batch, text_len, 2*alignment_dim)

        # Learned temperature (constrained to be positive)
        tau = torch.exp(self.log_tau) + 1e-6

        # Compute value and gate streams
        v = self.val(x_in)          # (batch, text_len, alignment_dim * n_glu_heads)
        g = self.gate(x_in) / tau   # (batch, text_len, alignment_dim * n_glu_heads)

        # Reshape for multi-head processing
        H, D = self.n_glu_heads, self.alignment_dim
        v = v.view(batch_size, text_len, H, D)  # (batch, text_len, n_heads, alignment_dim)
        g = g.view(batch_size, text_len, H, D)

        # GLU: gated = sigmoid(gate) * value
        gated = torch.sigmoid(g) * v  # (batch, text_len, n_heads, alignment_dim)

        # Reshape back and project to single score
        gated = gated.reshape(batch_size, text_len, H * D)  # (batch, text_len, alignment_dim * n_heads)
        alignment_scores = self.proj(gated).squeeze(-1)     # (batch, text_len)

        # Mask out padding tokens
        if text_attention_mask is not None:
            alignment_scores = alignment_scores * text_attention_mask

        return aligned_representations, alignment_scores, alignment_matrix
