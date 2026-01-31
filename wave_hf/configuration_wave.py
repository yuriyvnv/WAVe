"""
WAVe model configuration.

This module contains the configuration class for WAVe models.
"""

from transformers import PretrainedConfig
from typing import Optional


class WAVeConfig(PretrainedConfig):
    r"""
    Configuration class for WAVe (Word-Aligned Verification) model.

    This is the configuration class to store the configuration of a [`WAVeModel`].
    It is used to instantiate a WAVe model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        text_model_name_or_path (`str`, *optional*, defaults to `"sentence-transformers/all-roberta-large-v1"`):
            Path to pretrained text encoder model or model identifier from huggingface.co/models.
        audio_model_name_or_path (`str`, *optional*, defaults to `"facebook/w2v-bert-2.0"`):
            Path to pretrained audio encoder model or model identifier from huggingface.co/models.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the text and audio projection layers (shared embedding space).
        text_embedding_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the text encoder output.
        audio_embedding_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of the audio encoder output.
        alignment_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the word-level alignment module.
        num_alignment_heads (`int`, *optional*, defaults to 6):
            Number of attention heads in the word-level alignment module.
        n_glu_heads (`int`, *optional*, defaults to 4):
            Number of GLU heads for the alignment quality scorer.
        use_word_alignment (`bool`, *optional*, defaults to `True`):
            Whether to use word-level alignment module. This is the core innovation of WAVe.
        use_cross_modal (`bool`, *optional*, defaults to `False`):
            Whether to use cross-modal attention between text and audio sequences.
            Note: Default trained models do NOT use this feature.
        use_attentive_pooling (`bool`, *optional*, defaults to `False`):
            Whether to use attentive pooling instead of CLS token/mean pooling.
            Note: Default trained models do NOT use this feature.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for all fully connected layers in the model.
        freeze_encoders (`str`, *optional*, defaults to `"partial"`):
            Encoder freezing strategy. Can be:
            - "full": Freeze all encoder parameters
            - "partial": Freeze all but last N layers
            - "none": Keep all parameters trainable
        text_layers_to_unfreeze (`int`, *optional*, defaults to 3):
            Number of text encoder layers to unfreeze when `freeze_encoders="partial"`.
        audio_layers_to_unfreeze (`int`, *optional*, defaults to 3):
            Number of audio encoder layers to unfreeze when `freeze_encoders="partial"`.
        temperature (`float`, *optional*, defaults to 0.1):
            Temperature parameter for InfoNCE contrastive loss.
        projection_hidden_dim (`int`, *optional*):
            Hidden dimension for projection layers. If None, defaults to `projection_dim * 2`.
        projection_activation (`str`, *optional*, defaults to `"gelu"`):
            Activation function for projection layers. Can be "gelu" or "relu".
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.

    Example:
        ```python
        >>> from wave_hf import WAVeConfig, WAVeModel
        >>>
        >>> # Initializing a WAVe configuration matching the trained models
        >>> configuration = WAVeConfig()
        >>>
        >>> # Initializing a model from the configuration
        >>> model = WAVeModel(configuration)
        >>>
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "wave"

    def __init__(
        self,
        text_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        audio_model_name_or_path: str = "facebook/w2v-bert-2.0",
        projection_dim: int = 768,
        text_embedding_dim: int = 768,
        audio_embedding_dim: int = 1024,
        alignment_dim: int = 768,
        num_alignment_heads: int = 6,
        n_glu_heads: int = 4,
        use_word_alignment: bool = True,   # DEFAULT: Enabled (core innovation)
        use_cross_modal: bool = False,      # DEFAULT: Disabled (not used in training)
        use_attentive_pooling: bool = False,  # DEFAULT: Disabled (not used in training)
        dropout: float = 0.1,
        freeze_encoders: str = "partial",
        text_layers_to_unfreeze: int = 3,
        audio_layers_to_unfreeze: int = 3,
        temperature: float = 0.1,
        projection_hidden_dim: Optional[int] = None,
        projection_activation: str = "gelu",
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.text_model_name_or_path = text_model_name_or_path
        self.audio_model_name_or_path = audio_model_name_or_path
        self.projection_dim = projection_dim
        self.text_embedding_dim = text_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        self.alignment_dim = alignment_dim
        self.num_alignment_heads = num_alignment_heads
        self.n_glu_heads = n_glu_heads
        self.use_word_alignment = use_word_alignment
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        self.dropout = dropout
        self.freeze_encoders = freeze_encoders
        self.text_layers_to_unfreeze = text_layers_to_unfreeze
        self.audio_layers_to_unfreeze = audio_layers_to_unfreeze
        self.temperature = temperature
        self.projection_hidden_dim = projection_hidden_dim or projection_dim * 2
        self.projection_activation = projection_activation
        self.initializer_range = initializer_range
