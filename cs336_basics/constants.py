"""
Global constants for CS336 assignments.

This module contains constants used throughout the CS336 codebase, including
data file names, model configurations, and other shared values.

Usage:
    from cs336_basics.constants import DATA_FILES

    # Use constants instead of hardcoded strings
    data_path = get_data_path(DATA_FILES.TINYSTORIES_TRAIN)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


class DataFiles:
    """Constants for data files in the data/ directory."""

    TINYSTORIES_TRAIN: Final[str] = "TinyStoriesV2-GPT4-train.txt"
    """TinyStories training dataset (2.2GB)"""

    TINYSTORIES_VALID: Final[str] = "TinyStoriesV2-GPT4-valid.txt"
    """TinyStories validation dataset (22MB)"""

    OWT_TRAIN: Final[str] = "owt_train.txt"
    """OpenWebText training dataset (11.9GB)"""

    OWT_VALID: Final[str] = "owt_valid.txt"
    """OpenWebText validation dataset (290MB)"""

    # Alternative naming for convenience
    TINY_TRAIN: Final[str] = TINYSTORIES_TRAIN
    """TinyStories training dataset (2.2GB) - alternative name"""

    TINY_VALID: Final[str] = TINYSTORIES_VALID
    """TinyStories validation dataset (22MB) - alternative name"""

    OPENWEBTEXT_TRAIN: Final[str] = OWT_TRAIN
    """OpenWebText training dataset (11.9GB) - alternative name"""

    OPENWEBTEXT_VALID: Final[str] = OWT_VALID
    """OpenWebText validation dataset (290MB) - alternative name"""


class ModelDefaults:
    """Default model configuration constants."""

    # Tokenizer defaults
    DEFAULT_VOCAB_SIZE: Final[int] = 50257
    """GPT-2 vocabulary size"""

    DEFAULT_SPECIAL_TOKENS: Final[list[str]] = ["<|endoftext|>"]
    """Default special tokens for tokenizer"""

    # Model architecture defaults
    DEFAULT_D_MODEL: Final[int] = 768
    """Model dimension (embedding size)"""

    DEFAULT_NUM_HEADS: Final[int] = 12
    """Number of attention heads"""

    DEFAULT_NUM_LAYERS: Final[int] = 12
    """Number of transformer layers"""

    DEFAULT_D_FF: Final[int] = 3072
    """Feed-forward dimension"""

    DEFAULT_CONTEXT_LENGTH: Final[int] = 1024
    """Maximum sequence length"""

    # Training defaults
    DEFAULT_BATCH_SIZE: Final[int] = 32
    """Default batch size for training"""

    DEFAULT_LEARNING_RATE: Final[float] = 3e-4
    """Default learning rate"""

    DEFAULT_WARMUP_STEPS: Final[int] = 1000
    """Number of warmup steps for learning rate schedule"""


class Patterns:
    """Regex patterns and other string constants."""

    # GPT-2 pre-tokenization pattern
    GPT2_PAT_STR: Final[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
    """GPT-2 pre-tokenization regex pattern"""

    GPT5_PAT_STR: Final[str] = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
    """GPT-5 pre-tokenization regex pattern"""

    # Special tokens
    END_OF_TEXT: Final[bytes] = b"<|endoftext|>"
    """End of text token (bytes)"""

    END_OF_TEXT_STR: Final[str] = "<|endoftext|>"
    """End of text token (string)"""


class Paths:
    """Path-related constants."""

    # Directory names
    DATA_DIR: Final[str] = "data"
    """Data directory name"""

    CHECKPOINTS_DIR: Final[str] = "checkpoints"
    """Checkpoints directory name"""

    LOGS_DIR: Final[str] = "logs"
    """Logs directory name"""

    # File extensions
    CHECKPOINT_EXT: Final[str] = ".pt"
    """Checkpoint file extension (PyTorch)"""

    LOG_EXT: Final[str] = ".log"
    """Log file extension"""

    JSON_EXT: Final[str] = ".json"
    """JSON file extension"""


# Create instances for easy access
DATA_FILES = DataFiles()
MODEL_DEFAULTS = ModelDefaults()
PATTERNS = Patterns()
PATHS = Paths()

# Export commonly used constants at module level for convenience
TINYSTORIES_TRAIN = DATA_FILES.TINYSTORIES_TRAIN
"""TinyStories training dataset (2.2GB)"""

TINYSTORIES_VALID = DATA_FILES.TINYSTORIES_VALID
"""TinyStories validation dataset (22MB)"""

OWT_TRAIN = DATA_FILES.OWT_TRAIN
"""OpenWebText training dataset (11.9GB)"""

OWT_VALID = DATA_FILES.OWT_VALID
"""OpenWebText validation dataset (290MB)"""

GPT2_PAT_STR = PATTERNS.GPT2_PAT_STR
"""GPT-2 pre-tokenization regex pattern"""

GPT5_PAT_STR = PATTERNS.GPT5_PAT_STR
"""GPT-5 pre-tokenization regex pattern"""

END_OF_TEXT_TOKEN = PATTERNS.END_OF_TEXT
"""End of text token (bytes)"""