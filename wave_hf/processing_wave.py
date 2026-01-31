"""
Processor for WAVe model.

This module contains the processor that combines text tokenization and audio feature extraction.
"""

from transformers import ProcessorMixin, AutoTokenizer, AutoFeatureExtractor
from typing import List, Optional, Union
import numpy as np
import torch


class WAVeProcessor(ProcessorMixin):
    """
    Constructs a WAVe processor which wraps a text tokenizer and audio feature extractor into a single processor.

    [`WAVeProcessor`] offers all the functionalities of [`AutoTokenizer`] and [`AutoFeatureExtractor`].
    See the docstring of [`~WAVeProcessor.__call__`] and [`~WAVeProcessor.decode`] for more information.

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
        feature_extractor ([`FeatureExtractionMixin`]):
            An instance of [`FeatureExtractionMixin`]. The feature extractor is a required input.

    Example:
        ```python
        >>> from wave_hf import WAVeProcessor
        >>> from transformers import AutoTokenizer, AutoFeatureExtractor
        >>>
        >>> # Create processor from scratch
        >>> tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        >>> processor = WAVeProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        >>>
        >>> # Or load directly
        >>> processor = WAVeProcessor.from_pretrained("yuriyvnv/wave-portuguese")
        >>>
        >>> # Process single example
        >>> text = "Olá, como você está?"
        >>> audio = np.random.randn(16000)  # 1 second at 16kHz
        >>> inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
        >>>
        >>> # Process batch
        >>> texts = ["Olá", "Como vai?"]
        >>> audios = [np.random.randn(16000), np.random.randn(24000)]
        >>> inputs = processor(text=texts, audio=audios, sampling_rate=16000, padding=True, return_tensors="pt")
        ```
    """

    attributes = ["tokenizer", "feature_extractor"]
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "AutoFeatureExtractor"

    def __init__(self, tokenizer, feature_extractor):
        super().__init__(tokenizer, feature_extractor)
        self.current_processor = self.tokenizer

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]] = None,
        sampling_rate: Optional[int] = 16000,
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        **kwargs
    ) -> dict:
        """
        Main method to prepare text and audio for the model.

        Args:
            text (`str`, `List[str]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string.
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where
                C is a number of channels, and T the sample length of the audio.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the audio waveform in Hz.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:
                - `True` or `'longest'`: Pad to the longest sequence in the batch.
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length`.
                - `False` or `'do_not_pad'` (default): No padding.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length.
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to the text encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the text encoder.
            - **input_values** -- Audio input values to be fed to the audio encoder.
            - **audio_attention_mask** -- List of indices specifying which audio frames should be attended to (if padding is used).

        Example:
            ```python
            >>> from wave_hf import WAVeProcessor
            >>> import numpy as np
            >>>
            >>> processor = WAVeProcessor.from_pretrained("yuriyvnv/wave-portuguese")
            >>>
            >>> # Single example
            >>> text = "Olá mundo"
            >>> audio = np.random.randn(16000)
            >>> inputs = processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
            >>>
            >>> # Batch
            >>> texts = ["Texto um", "Texto dois"]
            >>> audios = [np.random.randn(16000), np.random.randn(32000)]
            >>> inputs = processor(text=texts, audio=audios, sampling_rate=16000, padding=True, return_tensors="pt")
            ```
        """
        if text is None and audio is None:
            raise ValueError("You must provide either text or audio inputs, or both.")

        # Initialize output dictionary
        encoded_inputs = {}

        # ===== PROCESS TEXT =====
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                **kwargs
            )
            encoded_inputs.update(text_inputs)

        # ===== PROCESS AUDIO =====
        if audio is not None:
            # Handle different audio input formats
            if isinstance(audio, (np.ndarray, torch.Tensor)):
                # Single audio - check dimensionality
                if len(audio.shape) == 1:
                    # Single channel, convert to batch of 1
                    audio = [audio]
                elif len(audio.shape) == 2:
                    # Could be batched (batch_size, n_samples) or multi-channel (n_channels, n_samples)
                    # Assume it's batched
                    audio = list(audio)

            # Process audio features
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs
            )

            # Map audio feature extractor output keys to model input keys
            # Different audio models use different key names
            if "input_values" in audio_inputs:
                encoded_inputs["input_values"] = audio_inputs["input_values"]
            elif "input_features" in audio_inputs:
                # Some models use input_features instead of input_values
                encoded_inputs["input_values"] = audio_inputs["input_features"]

            # Add audio attention mask if present (created by padding)
            if "attention_mask" in audio_inputs:
                encoded_inputs["audio_attention_mask"] = audio_inputs["attention_mask"]

        return encoded_inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a [`WAVeProcessor`] from a pretrained processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - A string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a processor saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.

        Returns:
            [`WAVeProcessor`]: A WAVeProcessor object.

        Example:
            ```python
            >>> from wave_hf import WAVeProcessor
            >>>
            >>> # Load from HuggingFace Hub
            >>> processor = WAVeProcessor.from_pretrained("yuriyvnv/wave-portuguese")
            >>>
            >>> # Or load from local directory
            >>> processor = WAVeProcessor.from_pretrained("./my_saved_model")
            ```
        """
        try:
            # Try to load from saved processor files
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            # Fallback: load from config and create new processor
            from .configuration_wave import WAVeConfig

            config = WAVeConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

            tokenizer = AutoTokenizer.from_pretrained(
                config.text_model_name_or_path,
                **kwargs
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                config.audio_model_name_or_path,
                **kwargs
            )

            return cls(tokenizer=tokenizer, feature_extractor=feature_extractor)
