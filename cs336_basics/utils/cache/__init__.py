"""
Cache utilities for CS336 assignments.

This package provides generic and domain-specific caching functionality for
efficient data processing and training workflows.

Modules:
- base: Generic JSONCache base class for any cacheable data
- word_counts: Domain-specific cache for BPE tokenizer word frequency counts
- pair_counts: Domain-specific cache for BPE tokenizer byte pair frequency counts
- merges: Domain-specific cache for BPE tokenizer merge operations
- vocab: Domain-specific cache for BPE tokenizer vocabulary
- checkpoint: Checkpoint manager for incremental resumable BPE training

Public API:
- JSONCache: Universal base class for JSON file caching
- WordCountsCache: Specialized cache for word counts with hex-encoded byte tuples
- PairCountsCache: Specialized cache for byte pair counts with hex-encoded pairs
- MergesCache: Specialized cache for BPE merge sequences with vocab_size keys
- VocabCache: Specialized cache for vocabulary mappings with vocab_size keys
- CheckpointManager: Manager for incremental training checkpoints (reusable across vocab sizes)
"""

from .base import JSONCache
from .word_counts import WordCountsCache
from .pair_counts import PairCountsCache
from .merges import MergesCache
from .vocab import VocabCache
from .checkpoint import CheckpointManager

__all__ = ['JSONCache', 'WordCountsCache', 'PairCountsCache', 'MergesCache', 'VocabCache', 'CheckpointManager']