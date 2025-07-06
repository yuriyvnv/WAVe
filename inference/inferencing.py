#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio-Text Similarity Inference Script
This script loads a trained audio-text embedding model and computes similarity scores
between audio files and text transcripts.

IMPORTANT: This version exactly matches the training implementation from trainer_unfreeze.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
import argparse
import logging
from typing import Union, List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ===== MODEL COMPONENTS (EXACT COPY FROM TRAINING) =====

class EnhancedProjection(nn.Module):
    """
    Enhanced projection layer with multiple linear transformations and non-linearities.
    """
    def __init__(
        self,
        input_dim, 
        projection_dim, 
        hidden_dim=None, 
        dropout=0.1,
        activation="gelu"
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = projection_dim * 2
        
        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to capture relationships between modalities.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention components
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialization
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, context, attention_mask=None):
        """
        Apply cross-modal attention from x to context.
        
        Args:
            x: Query tensor [batch_size, seq_len_q, dim]
            context: Key/Value tensor [batch_size, seq_len_kv, dim]
            attention_mask: Optional mask [batch_size, seq_len_q, seq_len_kv]
        
        Returns:
            Tensor with same shape as x after attention with context
        """
        batch_size = x.shape[0]
        
        # Project and reshape for multi-head attention
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len_kv]
            # Convert 0s to -inf before softmax
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class AttentivePooling(nn.Module):
    """
    Attentive pooling for sequence data.
    Computes a weighted average of the sequence based on learned attention weights.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply attentive pooling to sequence.
        
        Args:
            hidden_states: Sequence to pool [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len] (1=keep, 0=mask)
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Calculate attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len] 
            hidden_states                    # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return pooled_output


class WordLevelAlignmentModule(nn.Module):
    """
    Module to align word-level representations from text with temporal segments in audio.
    Uses attention mechanism to create a soft alignment between words and audio frames.
    """
    def __init__(self, text_hidden_dim, audio_hidden_dim, alignment_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.text_hidden_dim = text_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.alignment_dim = alignment_dim
        self.num_heads = num_heads
        
        # Projection layers to create query (text) and key/value (audio) representations
        self.text_projection = nn.Linear(text_hidden_dim, alignment_dim)
        self.audio_projection = nn.Linear(audio_hidden_dim, alignment_dim)
        
        # Multi-head attention for alignment
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=alignment_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection and layer norm
        self.output_projection = nn.Linear(alignment_dim, alignment_dim)
        self.layer_norm = nn.LayerNorm(alignment_dim)
        
        # Alignment confidence scorer (predicts how well each word aligns with audio)
        self.alignment_confidence = nn.Sequential(
            nn.Linear(alignment_dim, alignment_dim // 2),
            nn.ReLU(),
            nn.Linear(alignment_dim // 2, 1)
        )
    
    def forward(self, text_hidden_states, audio_hidden_states, 
                text_attention_mask=None, audio_attention_mask=None):
        """
        Compute alignment between text words and audio frames.
        
        Args:
            text_hidden_states: Text token representations [batch_size, text_len, hidden_dim]
            audio_hidden_states: Audio frame representations [batch_size, audio_len, hidden_dim]
            text_attention_mask: Text mask [batch_size, text_len]
            audio_attention_mask: Audio mask [batch_size, audio_len]
            
        Returns:
            aligned_representations: Text representations aligned with audio [batch_size, text_len, hidden_dim]
            alignment_scores: Word-level alignment scores [batch_size, text_len]
            alignment_matrix: Full alignment matrix [batch_size, text_len, audio_len]
        """
        batch_size, text_len, _ = text_hidden_states.shape
        _, audio_len, _ = audio_hidden_states.shape
        
        # Project text and audio
        text_proj = self.text_projection(text_hidden_states)
        audio_proj = self.audio_projection(audio_hidden_states)
        
        # Convert masks to format needed by MultiheadAttention
        if text_attention_mask is not None:
            text_key_padding_mask = (1.0 - text_attention_mask).bool()
        else:
            text_key_padding_mask = None
            
        if audio_attention_mask is not None:
            audio_key_padding_mask = (1.0 - audio_attention_mask).bool()
        else:
            audio_key_padding_mask = None
        
        # Compute text-to-audio attention (words attending to relevant audio frames)
        aligned_representations, alignment_weights = self.alignment_attention(
            query=text_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=audio_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Return all attention heads
        )
        
        # Average attention weights across heads to get final alignment matrix
        alignment_matrix = alignment_weights.mean(dim=1)
        
        # Apply residual connection and layer norm
        aligned_representations = self.layer_norm(
            text_hidden_states + self.output_projection(aligned_representations)
        )
        
        # Compute confidence score for each word alignment
        # Higher score = more confident that the word aligns with some part of the audio
        alignment_scores = self.alignment_confidence(aligned_representations).squeeze(-1)
        
        # Mask out padding tokens
        if text_attention_mask is not None:
            alignment_scores = alignment_scores * text_attention_mask
        
        return aligned_representations, alignment_scores, alignment_matrix


class EnhancedAudioTextModel(nn.Module):
    """
    Enhanced Audio-Text multimodal embedding model with:
    - Improved projection layers
    - Cross-modal attention
    - Attentive pooling
    - Word-level alignment
    - Partial encoder unfreezing support
    """
    def __init__(
        self,
        text_model_name="sentence-transformers/all-roberta-large-v1",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=768,
        text_embedding_dim=768,
        audio_embedding_dim=1024,
        dropout=0.1,
        use_cross_modal=True,
        use_attentive_pooling=True,
        use_word_alignment=True,  # New parameter
        freeze_encoders="partial",    # Changed to string: "full", "partial", "none"
        text_layers_to_unfreeze=5,    # New parameter for partial unfreezing
        audio_layers_to_unfreeze=5,   # New parameter for partial unfreezing
    ):
        super().__init__()
        
        # Load pre-trained models
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        self.audio_hidden_dim = self.audio_encoder.config.hidden_size
        logger.info(f"Text encoder hidden dim: {self.text_hidden_dim}")
        logger.info(f"Audio encoder hidden dim: {self.audio_hidden_dim}")
        
        # Save configuration
        self.projection_dim = projection_dim
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        self.use_word_alignment = use_word_alignment
        
        # Note: freeze_encoders parameters are only used during training
        # They don't affect inference behavior
        
        # Enhanced projection heads
        self.text_projection = EnhancedProjection(
            input_dim=text_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        
        self.audio_projection = EnhancedProjection(
            input_dim=audio_embedding_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )
        
        # Cross-modal attention (optional)
        if use_cross_modal:
            self.text_seq_to_projection = nn.Linear(
                self.text_hidden_dim,  # Dynamic: actual text encoder hidden size
                projection_dim         # Target projection dimension
            )
            self.audio_seq_to_projection = nn.Linear(
                self.audio_hidden_dim, # Dynamic: actual audio encoder hidden size
                projection_dim         # Target projection dimension
            )

            self.text_to_audio_attention = CrossModalAttention(
                dim=projection_dim, 
                dropout=dropout
            )
            self.audio_to_text_attention = CrossModalAttention(
                dim=projection_dim, 
                dropout=dropout
            )
            
            # Fusion layers to combine original and cross-attended features
            self.text_fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.audio_fusion = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim)
            )
        
        # Attentive pooling (optional)
        if use_attentive_pooling:
            self.text_pooling = AttentivePooling(text_embedding_dim)
            self.audio_pooling = AttentivePooling(audio_embedding_dim)
        
        # Add word-level alignment module (NEW)
        if use_word_alignment:
            self.word_level_alignment = WordLevelAlignmentModule(
                text_hidden_dim=self.text_hidden_dim,      # 768 for RoBERTa
                audio_hidden_dim=self.audio_hidden_dim,    # 1024 for w2v-bert
                alignment_dim=projection_dim,              # 768 (your shared space)
                dropout=dropout
            )
    
    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode text inputs with enhanced processing.
        """
        # Get text embeddings from encoder
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Choose pooling method
        if self.use_attentive_pooling:
            # Apply attentive pooling
            text_embedding = self.text_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # Use CLS token embedding (BERT-style) or mean pooling
            text_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Project to shared space
        projected = self.text_projection(text_embedding)
        
        return projected, outputs.last_hidden_state
    
    def encode_audio(self, input_values, attention_mask=None):
        """
        Encode audio inputs with enhanced processing.
        """
        # Adapt input name based on the model's expected input
        try:
            # First try with input_values (older models)
            outputs = self.audio_encoder(
                input_values=input_values,
                attention_mask=attention_mask
            )
        except TypeError:
            try:
                # Then try with input_features (newer models)
                outputs = self.audio_encoder(
                    input_features=input_values,
                    attention_mask=attention_mask
                )
            except TypeError as e:
                # Fallback to a generic approach
                logger.warning(f"Using fallback approach for audio encoder: {e}")
                outputs = self.audio_encoder(input_values)
        
        # Handle different output formats
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Assume the first element is the hidden states
            hidden_states = outputs[0]
        
        # Choose pooling method
        if self.use_attentive_pooling:
            # Apply attentive pooling
            audio_embedding = self.audio_pooling(hidden_states, attention_mask)
        else:
            # Apply masking for proper mean calculation
            if attention_mask is not None:
                # Expand mask for feature dimension
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                # Mask the hidden states
                masked_hidden = hidden_states * mask_expanded
                # Sum and normalize by the number of actual tokens
                sum_embeddings = torch.sum(masked_hidden, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                audio_embedding = sum_embeddings / sum_mask
            else:
                # If no mask, just average over time dimension
                audio_embedding = torch.mean(hidden_states, dim=1)
        
        # Project to shared space
        projected = self.audio_projection(audio_embedding)
        
        return projected, hidden_states
    
    def apply_cross_modal_attention(self, 
                                   text_projected, text_hidden, text_mask,
                                   audio_projected, audio_hidden, audio_mask):
        """
        Apply cross-modal attention between text and audio.
        """
        if not self.use_cross_modal:
            return text_projected, audio_projected
    
        audio_proj_seq = self.audio_seq_to_projection(audio_hidden)  
        text_proj_seq = self.text_seq_to_projection(text_hidden)  

        try:
            text_attended = self.text_to_audio_attention(
                text_projected.unsqueeze(1),
                audio_proj_seq,
                audio_mask
            ).squeeze(1)
        except Exception as e:
            print(f"Error in text_to_audio_attention: {e}")
            raise
        
        try:
            audio_attended = self.audio_to_text_attention(
                audio_projected.unsqueeze(1),
                text_proj_seq,
                text_mask
            ).squeeze(1)
        except Exception as e:
            print(f"Error in audio_to_text_attention: {e}")
            raise
    
        
        # Fuse - now both are projection_dim size
        text_fused = self.text_fusion(torch.cat([text_projected, text_attended], dim=1))
        audio_fused = self.audio_fusion(torch.cat([audio_projected, audio_attended], dim=1))
        
        return text_fused, audio_fused


# ===== INFERENCE CLASS =====

class AudioTextSimilarityInference:
    """Class for computing similarity between audio and text using trained model."""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the inference model.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.pt file)
            device: Device to run inference on (cuda/cpu). If None, auto-detect.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration from saved state
        # Note: The training script saves config in the checkpoint
        config = checkpoint.get('config', {})
        
        # Get actual model names from bash script or use defaults
        text_model_name = config.get('text_model_name', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        audio_model_name = config.get('audio_model_name', 'facebook/w2v-bert-2.0')
        
        # Initialize model with saved configuration
        self.model = EnhancedAudioTextModel(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            projection_dim=config.get('projection_dim', 768),
            use_cross_modal=config.get('use_cross_modal', True),
            use_attentive_pooling=config.get('use_attentive_pooling', True),
            use_word_alignment=config.get('use_word_alignment', True)
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer and feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
        
        self.max_text_length = config.get('max_text_length', 128)
        logger.info("Model loaded successfully!")
        logger.info(f"Model configuration: {config}")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load and preprocess audio file."""
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio
    
    def compute_similarity(self, audio_path: str, text: str) -> float:
        """
        Compute similarity between an audio file and text transcript.
        This follows the exact same processing pipeline as training.
        
        Args:
            audio_path: Path to audio file
            text: Text transcript
            
        Returns:
            Similarity score between 0 and 1
        """
        # Load and process audio
        audio_array = self.load_audio(audio_path)
        
        # Process audio
        audio_features = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Process text
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        audio_inputs = {k: v.to(self.device) for k, v in audio_features.items()}
        text_inputs = {k: v.to(self.device) for k, v in text_encoding.items()}
        
        # Compute embeddings following the exact training pipeline
        with torch.no_grad():
            # Step 1: Encode text
            text_proj, text_hidden = self.model.encode_text(
                text_inputs['input_ids'],
                text_inputs['attention_mask']
            )
            
            # Step 2: Encode audio
            if 'input_features' in audio_inputs:
                audio_values = audio_inputs['input_features']
            else:
                audio_values = audio_inputs['input_values']
            
            audio_proj, audio_hidden = self.model.encode_audio(
                audio_values,
                audio_inputs.get('attention_mask', None)
            )
            
            # Step 3: Apply cross-modal attention if enabled
            if self.model.use_cross_modal:
                text_fused, audio_fused = self.model.apply_cross_modal_attention(
                    text_proj, text_hidden, text_inputs['attention_mask'],
                    audio_proj, audio_hidden, audio_inputs.get('attention_mask', None)
                )
            else:
                text_fused = text_proj
                audio_fused = audio_proj
            
            # Step 4: Apply word-level alignment if enabled (optional for similarity)
            if self.model.use_word_alignment:
                _, alignment_scores, _ = self.model.word_level_alignment(
                    text_hidden_states=text_hidden,
                    audio_hidden_states=audio_hidden,
                    text_attention_mask=text_inputs['attention_mask'],
                    audio_attention_mask=audio_inputs.get('attention_mask', None)
                )
            
            # Step 5: Normalize embeddings (CRITICAL - same as training)
            text_norm = F.normalize(text_fused, p=2, dim=1)
            audio_norm = F.normalize(audio_fused, p=2, dim=1)
            
            # Step 6: Compute cosine similarity
            similarity = (text_norm * audio_norm).sum(dim=1).item()
            
            # Step 7: Convert to [0, 1] range
            # Training used similarities in [-1, 1] range with temperature scaling
            # For inference, we convert to [0, 1] for interpretability
            similarity_scaled = (similarity + 1) / 2
            
        return similarity_scaled
    
    def compute_batch_similarity(self, audio_paths: List[str], texts: List[str]) -> List[float]:
        """Compute similarities for multiple audio-text pairs."""
        similarities = []
        for audio_path, text in zip(audio_paths, texts):
            sim = self.compute_similarity(audio_path, text)
            similarities.append(sim)
        return similarities


# ===== UTILITY FUNCTION =====

def to_human_readable(similarity: float) -> float:
    """Convert similarity to human-readable format (matching training)."""
    # The training code uses this conversion
    return (similarity + 1) / 2


# ===== MAIN FUNCTION =====

def main():
    parser = argparse.ArgumentParser(description="Audio-Text Similarity Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--text", type=str, required=True,
                        help="Text transcript")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize inference model
    inference = AudioTextSimilarityInference(args.checkpoint, args.device)
    
    # Compute similarity
    similarity = inference.compute_similarity(args.audio, args.text)
    
    print(f"\nAudio-Text Similarity Results:")
    print(f"Audio: {args.audio}")
    print(f"Text: {args.text}")
    print(f"Similarity Score: {similarity:.4f}")
    print(f"Interpretation: {'High' if similarity > 0.8 else 'Medium' if similarity > 0.6 else 'Low'} similarity")


if __name__ == "__main__":
    main()