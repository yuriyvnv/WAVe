"""
PyTorch WAVe model.

This module contains the main WAVe model implementation for HuggingFace Transformers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import ModelOutput

try:
    from .configuration_wave import WAVeConfig
    from .modules import (
        EnhancedProjection,
        CrossModalAttention,
        AttentivePooling,
        WordLevelAlignmentModule
    )
except ImportError:
    from configuration_wave import WAVeConfig
    from modules import (
        EnhancedProjection,
        CrossModalAttention,
        AttentivePooling,
        WordLevelAlignmentModule
    )


# ===== MODEL OUTPUTS =====

@dataclass
class WAVeModelOutput(ModelOutput):
    """
    Output type of [`WAVeModel`].

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
            Text embeddings obtained by applying the projection layer to the pooled output
            of the text encoder.
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
            Audio embeddings obtained by applying the projection layer to the pooled output
            of the audio encoder.
        text_model_output (`BaseModelOutputWithPooling`, *optional*):
            The output of the text encoder model.
        audio_model_output (`BaseModelOutputWithPooling`, *optional*):
            The output of the audio encoder model.
        alignment_scores (`torch.FloatTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Per-word alignment quality scores. Only returned when `use_word_alignment=True`.
            Higher values indicate better alignment between the word and corresponding audio.
        alignment_matrix (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, audio_sequence_length)`, *optional*):
            Full attention-based alignment matrix showing how each text token attends to audio frames.
            Only returned when `use_word_alignment=True`.
        aligned_text_embeds (`torch.FloatTensor` of shape `(batch_size, projection_dim)`, *optional*):
            Text embeddings after word-level alignment with audio. Only returned when `use_word_alignment=True`.
    """

    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    text_model_output: Optional[object] = None
    audio_model_output: Optional[object] = None
    alignment_scores: Optional[torch.FloatTensor] = None
    alignment_matrix: Optional[torch.FloatTensor] = None
    aligned_text_embeds: Optional[torch.FloatTensor] = None


@dataclass
class WAVeForQualityOutput(ModelOutput):
    """
    Output type of [`WAVeForQualityVerification`] model.

    Args:
        quality_score (`torch.FloatTensor` of shape `(batch_size,)`):
            Overall quality score in range [0, 1] for each audio-text pair.
            Computed as (cosine_similarity + mean_alignment_score) / 2.
        cosine_similarity (`torch.FloatTensor` of shape `(batch_size,)`):
            Cosine similarity between text and audio embeddings, normalized to [0, 1].
        mean_alignment_score (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Average per-word alignment score for each sample. Only present if `use_word_alignment=True`.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
            Text embeddings.
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
            Audio embeddings.
        alignment_scores (`torch.FloatTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Per-word alignment quality scores.
        alignment_matrix (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, audio_sequence_length)`, *optional*):
            Full alignment matrix.
        text_model_output (`BaseModelOutputWithPooling`, *optional*):
            Text encoder output.
        audio_model_output (`BaseModelOutputWithPooling`, *optional*):
            Audio encoder output.
    """

    quality_score: torch.FloatTensor = None
    cosine_similarity: torch.FloatTensor = None
    mean_alignment_score: Optional[torch.FloatTensor] = None
    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    alignment_scores: Optional[torch.FloatTensor] = None
    alignment_matrix: Optional[torch.FloatTensor] = None
    text_model_output: Optional[object] = None
    audio_model_output: Optional[object] = None


# ===== BASE MODEL CLASS =====

class WAVePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = WAVeConfig
    base_model_prefix = "wave"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)


# ===== MAIN MODEL =====

class WAVeModel(WAVePreTrainedModel):
    """
    WAVe: Word-Aligned Verification of Synthetic Speech for ASR.

    This model produces multimodal embeddings for speech and text with word-level alignment.
    It consists of:
    - Text encoder (RoBERTa-based)
    - Audio encoder (Wav2Vec2-BERT-based)
    - Projection layers mapping to shared embedding space
    - Word-level alignment module (optional, enabled by default)
    - Cross-modal attention (optional, disabled by default)

    Args:
        config ([`WAVeConfig`]): Model configuration class with all the parameters of the model.

    Example:
        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch
        >>>
        >>> processor = AutoProcessor.from_pretrained("yuriyvnv/wave-portuguese")
        >>> model = AutoModel.from_pretrained("yuriyvnv/wave-portuguese")
        >>>
        >>> # Prepare inputs
        >>> text = "Olá, como você está?"
        >>> audio = torch.randn(16000)  # 1 second at 16kHz
        >>>
        >>> inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>>
        >>> # Access outputs
        >>> text_embeds = outputs.text_embeds           # Shape: [1, 768]
        >>> audio_embeds = outputs.audio_embeds         # Shape: [1, 768]
        >>> alignment_scores = outputs.alignment_scores # Shape: [1, num_words]
        >>> alignment_matrix = outputs.alignment_matrix # Shape: [1, num_words, num_frames]
        ```
    """

    def __init__(self, config: WAVeConfig):
        super().__init__(config)
        self.config = config

        # ===== ENCODERS =====
        # Load pretrained encoders directly (no wrapping)
        self.text_encoder = AutoModel.from_pretrained(
            config.text_model_name_or_path,
            add_pooling_layer=False  # We'll do our own pooling
        )
        self.audio_encoder = AutoModel.from_pretrained(
            config.audio_model_name_or_path
        )

        # Get actual hidden sizes from loaded models
        self.text_hidden_size = self.text_encoder.config.hidden_size
        self.audio_hidden_size = self.audio_encoder.config.hidden_size

        # ===== PROJECTION LAYERS =====
        self.text_projection = EnhancedProjection(
            input_dim=self.text_hidden_size,
            projection_dim=config.projection_dim,
            hidden_dim=config.projection_hidden_dim,
            dropout=config.dropout,
            activation=config.projection_activation
        )

        self.audio_projection = EnhancedProjection(
            input_dim=self.audio_hidden_size,
            projection_dim=config.projection_dim,
            hidden_dim=config.projection_hidden_dim,
            dropout=config.dropout,
            activation=config.projection_activation
        )

        # ===== OPTIONAL: CROSS-MODAL ATTENTION =====
        if config.use_cross_modal:
            # Project sequences to common dimension first
            self.text_seq_projection = nn.Linear(self.text_hidden_size, config.projection_dim)
            self.audio_seq_projection = nn.Linear(self.audio_hidden_size, config.projection_dim)

            self.text_to_audio_attention = CrossModalAttention(
                dim=config.projection_dim,
                num_heads=8,
                dropout=config.dropout
            )
            self.audio_to_text_attention = CrossModalAttention(
                dim=config.projection_dim,
                num_heads=8,
                dropout=config.dropout
            )

            # Fusion layers
            self.text_fusion = nn.Sequential(
                nn.Linear(config.projection_dim * 2, config.projection_dim),
                nn.LayerNorm(config.projection_dim)
            )
            self.audio_fusion = nn.Sequential(
                nn.Linear(config.projection_dim * 2, config.projection_dim),
                nn.LayerNorm(config.projection_dim)
            )

        # ===== OPTIONAL: ATTENTIVE POOLING =====
        if config.use_attentive_pooling:
            self.text_attentive_pooling = AttentivePooling(self.text_hidden_size)
            self.audio_attentive_pooling = AttentivePooling(self.audio_hidden_size)

        # ===== WORD-LEVEL ALIGNMENT MODULE =====
        if config.use_word_alignment:
            self.word_level_alignment = WordLevelAlignmentModule(
                text_hidden_dim=self.text_hidden_size,
                audio_hidden_dim=self.audio_hidden_size,
                alignment_dim=config.alignment_dim,
                num_heads=config.num_alignment_heads,
                dropout=config.dropout,
                n_glu_heads=config.n_glu_heads
            )

            # Separate projection for aligned text
            self.aligned_text_projection = EnhancedProjection(
                input_dim=self.text_hidden_size,  # Aligned representations maintain text_hidden_size
                projection_dim=config.projection_dim,
                hidden_dim=config.projection_hidden_dim,
                dropout=config.dropout,
                activation=config.projection_activation
            )

        # Apply encoder freezing
        self._freeze_encoders()

        # Initialize weights
        self.post_init()

    def _freeze_encoders(self):
        """Apply encoder freezing strategy based on config."""

        if self.config.freeze_encoders == "full":
            # Freeze all encoder parameters
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        elif self.config.freeze_encoders == "partial":
            # TEXT ENCODER: Unfreeze last N layers
            if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                total_text_layers = len(self.text_encoder.encoder.layer)

                # Freeze all first
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

                # Unfreeze last N layers
                layers_to_unfreeze = min(self.config.text_layers_to_unfreeze, total_text_layers)
                for i in range(total_text_layers - layers_to_unfreeze, total_text_layers):
                    for param in self.text_encoder.encoder.layer[i].parameters():
                        param.requires_grad = True

                # Unfreeze pooler if it exists
                if hasattr(self.text_encoder, 'pooler') and self.text_encoder.pooler is not None:
                    for param in self.text_encoder.pooler.parameters():
                        param.requires_grad = True

            # AUDIO ENCODER: Unfreeze last N layers
            if hasattr(self.audio_encoder, 'encoder') and hasattr(self.audio_encoder.encoder, 'layers'):
                total_audio_layers = len(self.audio_encoder.encoder.layers)

                # Freeze all first
                for param in self.audio_encoder.parameters():
                    param.requires_grad = False

                # Unfreeze last N layers
                layers_to_unfreeze = min(self.config.audio_layers_to_unfreeze, total_audio_layers)
                for i in range(total_audio_layers - layers_to_unfreeze, total_audio_layers):
                    for param in self.audio_encoder.encoder.layers[i].parameters():
                        param.requires_grad = True

                # Unfreeze feature projection
                if hasattr(self.audio_encoder, 'feature_projection'):
                    for param in self.audio_encoder.feature_projection.parameters():
                        param.requires_grad = True

        # If "none", keep all parameters trainable (default)

    def _masked_mean_pooling(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Apply masked mean pooling to sequence.

        Args:
            hidden_states: Shape (batch_size, sequence_length, hidden_size)
            attention_mask: Shape (batch_size, sequence_length)

        Returns:
            Pooled output of shape (batch_size, hidden_size)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Expand mask to hidden_size
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum with mask
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

        return sum_hidden / sum_mask

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Extract text features.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Text token IDs.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, projection_dim)`: Text embeddings.
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pool (use CLS token or attentive pooling)
        if self.config.use_attentive_pooling:
            text_pooled = self.text_attentive_pooling(
                text_outputs.last_hidden_state,
                attention_mask
            )
        else:
            text_pooled = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Project
        text_embeds = self.text_projection(text_pooled)

        return text_embeds

    def get_audio_features(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Extract audio features.

        Args:
            input_values (`torch.FloatTensor`):
                Audio features of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, feature_dim)`.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Audio attention mask.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, projection_dim)`: Audio embeddings.
        """
        # Encode audio
        audio_outputs = self.audio_encoder(
            input_features=input_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Pool (masked mean pooling or attentive pooling)
        if self.config.use_attentive_pooling:
            audio_pooled = self.audio_attentive_pooling(
                audio_outputs.last_hidden_state,
                attention_mask
            )
        else:
            # Masked mean pooling
            audio_pooled = self._masked_mean_pooling(
                audio_outputs.last_hidden_state,
                attention_mask
            )

        # Project
        audio_embeds = self.audio_projection(audio_pooled)

        return audio_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, WAVeModelOutput]:
        """
        Forward pass of WAVe model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input text tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            input_values (`torch.FloatTensor`, *optional*):
                Audio input features or raw waveform.
            audio_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask for audio input.
            return_dict (`bool`, *optional*):
                Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether to return hidden states from encoders.

        Returns:
            [`WAVeModelOutput`] or `tuple`: Model outputs containing embeddings and alignment scores.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ===== ENCODE TEXT =====
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_hidden_states = text_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # ===== ENCODE AUDIO =====
        audio_outputs = self.audio_encoder(
            input_features=input_values,
            attention_mask=audio_attention_mask,
            return_dict=True
        )
        audio_hidden_states = audio_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # ===== OPTIONAL: CROSS-MODAL ATTENTION =====
        if self.config.use_cross_modal:
            # Project sequences to common dimension
            text_seq = self.text_seq_projection(text_hidden_states)
            audio_seq = self.audio_seq_projection(audio_hidden_states)

            # Apply cross-attention
            text_attended = self.text_to_audio_attention(
                text_seq, audio_seq, audio_attention_mask
            )
            audio_attended = self.audio_to_text_attention(
                audio_seq, text_seq, attention_mask
            )

            # Fusion
            text_hidden_states = self.text_fusion(torch.cat([text_seq, text_attended], dim=-1))
            audio_hidden_states = self.audio_fusion(torch.cat([audio_seq, audio_attended], dim=-1))

        # ===== POOLING =====
        if self.config.use_attentive_pooling:
            text_pooled = self.text_attentive_pooling(text_hidden_states, attention_mask)
            audio_pooled = self.audio_attentive_pooling(audio_hidden_states, audio_attention_mask)
        else:
            # CLS token for text
            text_pooled = text_hidden_states[:, 0, :]
            # Masked mean for audio
            audio_pooled = self._masked_mean_pooling(audio_hidden_states, audio_attention_mask)

        # ===== PROJECTION =====
        text_embeds = self.text_projection(text_pooled)
        audio_embeds = self.audio_projection(audio_pooled)

        # ===== WORD-LEVEL ALIGNMENT =====
        alignment_scores = None
        alignment_matrix = None
        aligned_text_embeds = None

        if self.config.use_word_alignment:
            # Compute word-to-audio alignment
            aligned_repr, alignment_scores, alignment_matrix = self.word_level_alignment(
                text_hidden_states=text_hidden_states,
                audio_hidden_states=audio_hidden_states,
                text_attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask
            )

            # Pool aligned representations
            if self.config.use_attentive_pooling:
                aligned_pooled = self.text_attentive_pooling(aligned_repr, attention_mask)
            else:
                aligned_pooled = aligned_repr[:, 0, :]  # CLS token

            # Project aligned text
            aligned_text_embeds = self.aligned_text_projection(aligned_pooled)

        # ===== RETURN =====
        if not return_dict:
            return tuple(
                v for v in [
                    text_embeds,
                    audio_embeds,
                    text_outputs if output_hidden_states else None,
                    audio_outputs if output_hidden_states else None,
                    alignment_scores,
                    alignment_matrix,
                    aligned_text_embeds
                ] if v is not None
            )

        return WAVeModelOutput(
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs if output_hidden_states else None,
            audio_model_output=audio_outputs if output_hidden_states else None,
            alignment_scores=alignment_scores,
            alignment_matrix=alignment_matrix,
            aligned_text_embeds=aligned_text_embeds
        )


# ===== QUALITY VERIFICATION MODEL =====

class WAVeForQualityVerification(WAVePreTrainedModel):
    """
    WAVe model with quality scoring for synthetic speech verification.

    This model extends [`WAVeModel`] by computing a single quality score that combines:
    - Cosine similarity between text and audio embeddings (global semantic alignment)
    - Mean alignment score from word-level alignment module (local quality)

    The quality score is in range [0, 1] where:
    - 0.8-1.0: High quality (recommended for ASR training)
    - 0.5-0.8: Medium quality (may be useful with filtering)
    - 0.0-0.5: Low quality (likely has synthesis errors)

    Args:
        config ([`WAVeConfig`]): Model configuration class with all the parameters of the model.

    Example:
        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch
        >>>
        >>> processor = AutoProcessor.from_pretrained("yuriyvnv/wave-portuguese")
        >>> model = AutoModel.from_pretrained("yuriyvnv/wave-portuguese")
        >>>
        >>> # Prepare synthetic audio-text pair
        >>> text = "Olá, como você está?"
        >>> audio = torch.randn(16000)  # Synthetic audio
        >>>
        >>> inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>>
        >>> # Get quality score
        >>> quality = outputs.quality_score.item()  # e.g., 0.87
        >>>
        >>> # Filter based on threshold
        >>> if quality >= 0.8:
        ...     print("High quality - safe for ASR training")
        >>> elif quality >= 0.5:
        ...     print("Medium quality - use with caution")
        >>> else:
        ...     print("Low quality - discard")
        >>>
        >>> # Access detailed outputs
        >>> cosine_sim = outputs.cosine_similarity
        >>> per_word_scores = outputs.alignment_scores
        >>> embeddings = outputs.text_embeds
        ```
    """

    def __init__(self, config: WAVeConfig):
        super().__init__(config)
        self.config = config

        # Base WAVe model
        self.wave = WAVeModel(config)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        audio_attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple, WAVeForQualityOutput]:
        """
        Forward pass with quality score computation.

        Args:
            input_ids: Text token IDs of shape `(batch_size, sequence_length)`
            attention_mask: Text attention mask of shape `(batch_size, sequence_length)`
            input_values: Audio features
            audio_attention_mask: Audio attention mask of shape `(batch_size, sequence_length)`
            return_dict: Whether to return [`~utils.ModelOutput`] instead of tuple
            output_hidden_states: Whether to return encoder hidden states

        Returns:
            [`WAVeForQualityOutput`] with quality_score as primary output.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get base model outputs
        wave_outputs = self.wave(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            audio_attention_mask=audio_attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )

        # ===== COMPUTE COSINE SIMILARITY =====
        # Normalize embeddings
        text_embeds_norm = F.normalize(wave_outputs.text_embeds, p=2, dim=-1)
        audio_embeds_norm = F.normalize(wave_outputs.audio_embeds, p=2, dim=-1)

        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (text_embeds_norm * audio_embeds_norm).sum(dim=-1)

        # Normalize to [0, 1] range (from [-1, 1])
        cosine_sim_normalized = (cosine_sim + 1.0) / 2.0

        # ===== COMPUTE MEAN ALIGNMENT SCORE =====
        mean_alignment = None

        if wave_outputs.alignment_scores is not None:
            # Apply sigmoid to get scores in [0, 1] range
            alignment_scores_sigmoid = torch.sigmoid(wave_outputs.alignment_scores)

            # Compute mean per sample, accounting for padding
            if attention_mask is not None:
                # Mask out padding tokens
                masked_alignment = alignment_scores_sigmoid * attention_mask
                # Sum and normalize by actual sequence length
                mean_alignment = masked_alignment.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            else:
                # No mask, simple mean
                mean_alignment = alignment_scores_sigmoid.mean(dim=1)

        # ===== COMPUTE FINAL QUALITY SCORE =====
        if mean_alignment is not None:
            # Combine cosine similarity and alignment score
            quality_score = (cosine_sim_normalized + mean_alignment) / 2.0
        else:
            # No alignment available, use only cosine similarity
            quality_score = cosine_sim_normalized

        # ===== RETURN =====
        if not return_dict:
            return (
                quality_score,
                cosine_sim_normalized,
                mean_alignment,
                wave_outputs.text_embeds,
                wave_outputs.audio_embeds,
                wave_outputs.alignment_scores,
                wave_outputs.alignment_matrix,
            )

        return WAVeForQualityOutput(
            quality_score=quality_score,
            cosine_similarity=cosine_sim_normalized,
            mean_alignment_score=mean_alignment,
            text_embeds=wave_outputs.text_embeds,
            audio_embeds=wave_outputs.audio_embeds,
            alignment_scores=wave_outputs.alignment_scores,
            alignment_matrix=wave_outputs.alignment_matrix,
            text_model_output=wave_outputs.text_model_output,
            audio_model_output=wave_outputs.audio_model_output,
        )

    def verify_quality(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        input_values: torch.FloatTensor,
        audio_attention_mask: Optional[torch.FloatTensor] = None,
        threshold: float = 0.8,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Convenience method for batch quality verification with threshold.

        Args:
            input_ids: Text token IDs
            attention_mask: Text attention mask
            input_values: Audio features
            audio_attention_mask: Audio attention mask
            threshold: Quality threshold (default 0.8 for high quality)

        Returns:
            tuple containing:
                - quality_scores: Quality score for each sample in batch
                - is_high_quality: Boolean mask indicating samples above threshold

        Example:
            ```python
            >>> scores, high_quality_mask = model.verify_quality(
            ...     input_ids=batch["input_ids"],
            ...     attention_mask=batch["attention_mask"],
            ...     input_values=batch["audio"],
            ...     threshold=0.8
            ... )
            >>> # Filter high quality samples
            >>> high_quality_indices = high_quality_mask.nonzero(as_tuple=True)[0]
            ```
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            audio_attention_mask=audio_attention_mask,
            return_dict=True,
        )

        quality_scores = outputs.quality_score
        is_high_quality = quality_scores >= threshold

        return quality_scores, is_high_quality
