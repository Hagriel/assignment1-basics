"""
Utility functions for CS336 assignments.

This package contains common utility functions that are used across multiple
assignments and components, organized by functionality:

- time: Time formatting utilities
- path: Path resolution and management
- parallel: General parallel processing utilities
- file_processing: File chunking and multiprocessing for file operations
- cache: JSON-based caching system (base + domain-specific implementations)
- training_logger: Progress logging for training workflows
- wandb_logger: Weights & Biases integration for experiment tracking
- import_utils: Import setup utilities for scripts

"""

# Time utilities
from .time import format_time

# Path utilities
from .path import (
    get_project_root,
    add_project_to_path,
    setup_imports,
    get_data_path,
    ensure_data_directory,
    ensure_project_imports,
    temp_add_to_path,
)

# Parallel processing utilities
from .parallel import process_chunks_parallel

# File processing utilities
from .file_processing import (
    find_chunk_boundaries,
    process_file_chunks_multiprocessing,
)

# Cache utilities (exported from cache subpackage)
from .cache import JSONCache, WordCountsCache, PairCountsCache, MergesCache, VocabCache, CheckpointManager

# Training utilities
from .training_logger import TrainingLogger

# Wandb integration (optional - imports gracefully handled)
try:
    from .wandb_logger import WandbLogger, is_wandb_available
    _wandb_available = True
except ImportError:
    WandbLogger = None
    is_wandb_available = lambda: False
    _wandb_available = False

# Import utilities
from .import_utils import setup_imports_from_file


__all__ = [
    # Time
    'format_time',
    # Path
    'get_project_root',
    'add_project_to_path',
    'setup_imports',
    'get_data_path',
    'ensure_data_directory',
    'ensure_project_imports',
    'temp_add_to_path',
    # Parallel
    'process_chunks_parallel',
    # File processing
    'find_chunk_boundaries',
    'process_file_chunks_multiprocessing',
    # Cache
    'JSONCache',
    'WordCountsCache',
    'PairCountsCache',
    'MergesCache',
    'VocabCache',
    'CheckpointManager',
    # Training
    'TrainingLogger',
    # Wandb (optional)
    'WandbLogger',
    'is_wandb_available',
    # Import
    'setup_imports_from_file',
]