"""
WAVe model modules.

This package contains the core architectural components of the WAVe model.
"""

from .projection import EnhancedProjection
from .attention import CrossModalAttention
from .pooling import AttentivePooling
from .alignment import WordLevelAlignmentModule

__all__ = [
    "EnhancedProjection",
    "CrossModalAttention",
    "AttentivePooling",
    "WordLevelAlignmentModule",
]
