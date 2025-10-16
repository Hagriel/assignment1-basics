"""
Cache utilities for CS336 assignments.

This package provides generic and domain-specific caching functionality for
efficient data processing and training workflows.

Modules:
- base: Generic JSONCache base class for any cacheable data
- word_counts: Domain-specific cache for BPE tokenizer word frequency counts

Public API:
- JSONCache: Universal base class for JSON file caching
- WordCountsCache: Specialized cache for word counts with hex-encoded byte tuples
"""

from .base import JSONCache
from .word_counts import WordCountsCache

__all__ = ['JSONCache', 'WordCountsCache']