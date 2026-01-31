"""
Standalone version of EnhancedAudioTextModel for comparison.
This extracts just the model class without module-level CUDA checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Copy the necessary classes from trainer_unfreeze.py
# (Simplified versions without the problematic imports)

class EnhancedProjection(nn.Module):
    def __init__(self, input_dim, projection_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = projection_dim * 2

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, x):
        return self.projection(x)


class WordLevelAlignmentModule(nn.Module):
    def __init__(self, text_hidden_dim, audio_hidden_dim, alignment_dim,
                 num_heads=6, dropout=0.1, n_glu_heads=4):
        super().__init__()
        self.text_projection = nn.Linear(text_hidden_dim, alignment_dim)
        self.audio_projection = nn.Linear(audio_hidden_dim, alignment_dim)
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=alignment_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(alignment_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-head GLU scorer
        self.log_tau = nn.Parameter(torch.zeros(()))
        out_dim = alignment_dim * n_glu_heads
        self.val = nn.Linear(alignment_dim * 2, out_dim)
        self.gate = nn.Linear(alignment_dim * 2, out_dim)
        self.proj = nn.Linear(out_dim, 1)
        self.output_projection = nn.Linear(alignment_dim, alignment_dim)

    def forward(self, text_hidden_states, audio_hidden_states,
                text_attention_mask=None, audio_attention_mask=None):
        text_proj = self.text_projection(text_hidden_states)
        audio_proj = self.audio_projection(audio_hidden_states)

        aligned_representations, alignment_weights = self.alignment_attention(
            query=text_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=(1 - audio_attention_mask).bool() if audio_attention_mask is not None else None,
            need_weights=True,
            average_attn_weights=False
        )

        aligned_representations = self.dropout(aligned_representations)
        aligned_representations = self.layer_norm(text_proj + aligned_representations)

        # GLU scoring
        x_in = torch.cat([text_proj, aligned_representations], dim=-1)
        tau = torch.exp(self.log_tau) + 1e-6
        v = self.val(x_in)
        g = self.gate(x_in) / tau
        gated = torch.sigmoid(g) * v
        alignment_scores = self.proj(gated).squeeze(-1)

        final_representations = self.output_projection(aligned_representations)

        return {
            "aligned_representations": final_representations,
            "alignment_scores": alignment_scores,
            "alignment_matrix": alignment_weights.mean(dim=1)
        }


class EnhancedAudioTextModel(nn.Module):
    def __init__(
        self,
        text_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        audio_model_name="facebook/w2v-bert-2.0",
        projection_dim=768,
        text_embedding_dim=768,
        audio_embedding_dim=1024,
        dropout=0.1,
        use_cross_modal=False,
        use_attentive_pooling=False,
        use_word_alignment=True,
        freeze_encoders="none",
        text_layers_to_unfreeze=5,
        audio_layers_to_unfreeze=5,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size
        self.audio_hidden_dim = self.audio_encoder.config.hidden_size

        self.projection_dim = projection_dim
        self.use_cross_modal = use_cross_modal
        self.use_attentive_pooling = use_attentive_pooling
        self.use_word_alignment = use_word_alignment

        # Projection layers
        self.text_projection = EnhancedProjection(
            input_dim=self.text_hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )

        self.audio_projection = EnhancedProjection(
            input_dim=self.audio_hidden_dim,
            projection_dim=projection_dim,
            dropout=dropout
        )

        # Word-level alignment
        if use_word_alignment:
            self.word_level_alignment = WordLevelAlignmentModule(
                text_hidden_dim=self.text_hidden_dim,
                audio_hidden_dim=self.audio_hidden_dim,
                alignment_dim=projection_dim,
                num_heads=6,
                dropout=dropout,
                n_glu_heads=4
            )
            self.aligned_text_projection = EnhancedProjection(
                input_dim=self.text_hidden_dim,
                projection_dim=projection_dim,
                dropout=dropout
            )

    def forward(self, input_ids, attention_mask, input_values, audio_attention_mask=None):
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_hidden_states = text_outputs.last_hidden_state
        text_cls = text_hidden_states[:, 0, :]

        # Encode audio
        audio_outputs = self.audio_encoder(
            input_features=input_values,
            attention_mask=audio_attention_mask,
            return_dict=True
        )
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_pooled = audio_hidden_states.mean(dim=1)

        # Project
        text_embeds = self.text_projection(text_cls)
        audio_embeds = self.audio_projection(audio_pooled)

        # Normalize
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)

        outputs = {
            "text_embeds": text_embeds,
            "audio_embeds": audio_embeds,
        }

        # Word-level alignment
        if self.use_word_alignment:
            alignment_outputs = self.word_level_alignment(
                text_hidden_states=text_hidden_states,
                audio_hidden_states=audio_hidden_states,
                text_attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask
            )
            outputs["alignment_scores"] = alignment_outputs["alignment_scores"]
            outputs["alignment_matrix"] = alignment_outputs["alignment_matrix"]
            outputs["aligned_text_hidden_states"] = alignment_outputs["aligned_representations"]

            # Project aligned text
            aligned_pooled = alignment_outputs["aligned_representations"][:, 0, :]
            aligned_embeds = self.aligned_text_projection(aligned_pooled)
            aligned_embeds = F.normalize(aligned_embeds, p=2, dim=-1)
            outputs["aligned_text_embeds"] = aligned_embeds

        return outputs
