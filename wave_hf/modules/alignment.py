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
