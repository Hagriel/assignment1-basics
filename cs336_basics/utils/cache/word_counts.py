"""
Domain-specific cache implementation for BPE tokenizer word counts.

This module provides a specialized cache for storing and retrieving word frequency
counts used in BPE tokenizer training, with support for hex-encoded byte tuples
and frequency-based sorting.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING, Callable, Any

from .base import JSONCache

if TYPE_CHECKING:
    pass


class WordCountsCache(JSONCache[Counter[tuple[bytes, ...]]]):
    """
    Domain-specific cache for BPE tokenizer word frequency counts.

    Handles serialization of Counter[tuple[bytes, ...]] to JSON format with:
    - Hex-encoded byte tuples for human readability
    - Optional sorting by frequency with lexicographic tie-breaking
    - Integration with training logger for statistics
    """

    def __init__(self, cache_dir: Path | None = None, logger: Any | None = None) -> None:
        """
        Initialize word counts cache.

        Args:
            cache_dir: Directory for cache files (default: project_root/data)
            logger: Optional logger with log_stats method (default: None)
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        super().__init__(cache_dir=cache_dir, cache_prefix="word_counts", logger=logger)
        self._sort_by_frequency = True  # Default behavior for word counts

    def _get_default_cache_dir(self) -> Path:
        """Get the default cache directory (project_root/data/cache)."""
        from cs336_basics.utils import get_project_root
        return get_project_root() / "data" / "cache"

    def get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.

        Organizes cache files by dataset name in subdirectories.
        Creates: data/cache/{dataset_name}/word_counts.json

        Args:
            key: Cache key (can be filename or full path)

        Returns:
            Path to the cache file
        """
        # Extract just the filename if a full path was provided
        if '/' in key or '\\' in key:
            key = Path(key).name

        # Create subdirectory for this dataset
        dataset_dir = self.cache_dir / key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Return path without repeating the dataset name in filename
        return dataset_dir / f"{self.cache_prefix}.json"

    def serialize(self, data: Counter[tuple[bytes, ...]]) -> dict[str, Any]:
        """
        Serialize word counts to JSON format.

        Converts Counter[tuple[bytes, ...]] to dictionary with:
        - Keys: JSON arrays of byte strings (UTF-8 if possible, hex prefixed with "\\x" otherwise)
        - Values: frequency counts
        - Optionally sorted by frequency (descending) with lexicographic tie-breaking

        Args:
            data: Counter mapping word tuples to their frequencies

        Returns:
            JSON-serializable dictionary
        """
        serializable_data = {}

        # Determine iteration order (sorted by frequency or natural order)
        if self._sort_by_frequency:
            # Sort by frequency (descending), then lexicographically (ascending)
            items = sorted(data.items(), key=lambda x: (-x[1], x[0]))
        else:
            items = data.items()

        # Convert each word tuple to string array (UTF-8 or hex)
        for word_tuple, count in items:
            key_parts = []
            for b in word_tuple:
                try:
                    # Try to decode as UTF-8
                    decoded = b.decode('utf-8')
                    # Use the string directly if it's printable and doesn't need escaping
                    key_parts.append(decoded)
                except UnicodeDecodeError:
                    # Fall back to hex with \x prefix for non-UTF-8 bytes
                    key_parts.append(f"\\x{b.hex()}")

            key = json.dumps(key_parts)
            serializable_data[key] = count

        return serializable_data

    def deserialize(self, json_data: dict[str, Any]) -> Counter[tuple[bytes, ...]]:
        """
        Deserialize JSON data back to word counts.

        Converts dictionary with UTF-8/hex keys back to Counter[tuple[bytes, ...]].

        Args:
            json_data: Dictionary loaded from JSON file

        Returns:
            Counter mapping word tuples to frequencies

        Raises:
            ValueError: If data format is invalid
        """
        word_counts: Counter[tuple[bytes, ...]] = Counter()

        for key, count in json_data.items():
            # Parse the JSON array
            str_list = json.loads(key)
            word_tuple_parts = []

            for s in str_list:
                if isinstance(s, str) and s.startswith("\\x"):
                    # Hex-encoded bytes: "\x48656c6c6f" -> b'Hello'
                    hex_str = s[2:]  # Remove \x prefix
                    word_tuple_parts.append(bytes.fromhex(hex_str))
                else:
                    # UTF-8 string: encode back to bytes
                    word_tuple_parts.append(s.encode('utf-8'))

            word_counts[tuple(word_tuple_parts)] = count

        return word_counts

    def save(
        self,
        data: Counter[tuple[bytes, ...]],
        key: str,
        sort_by_frequency: bool = False,
        **kwargs: Any
    ) -> Path:
        """
        Save word counts to cache with frequency sorting option.

        Args:
            data: Word counts to save
            key: Cache key (typically source filename)
            sort_by_frequency: Whether to sort by frequency before saving
            **kwargs: Additional arguments passed to parent save()

        Returns:
            Path to saved cache file
        """
        # Set sorting preference before serialization
        self._sort_by_frequency = sort_by_frequency

        return super().save(data, key, **kwargs)

    def _log_load(self, cache_path: Path, data: Counter[tuple[bytes, ...]]) -> None:
        """Log statistics when loading word counts."""
        if self.logger and hasattr(self.logger, 'log_stats'):
            self.logger.log_stats(
                **{
                    "Loaded word counts from cache": "",
                    "Total unique words": len(data),
                    "Total word occurrences": sum(data.values())
                }
            )

    # Convenience methods for domain-specific operations
    def load_or_compute_with_callbacks(
        self,
        source_filename: str,
        compute_fn: Callable[[], Counter[tuple[bytes, ...]]],
        force_recompute: bool = False,
        sort_on_save: bool = False,
        on_load: Callable[[Counter[tuple[bytes, ...]]], None] | None = None,
        on_compute: Callable[[Counter[tuple[bytes, ...]]], None] | None = None,
        verbose: bool = False,
    ) -> Counter[tuple[bytes, ...]]:
        """
        Extended load_or_compute with callbacks for word counts.

        This method provides the same interface as the original implementation
        with support for custom callbacks and verbose logging.

        Args:
            source_filename: Name of the source file
            compute_fn: Function to compute word counts on cache miss
            force_recompute: If True, bypass cache and recompute
            sort_on_save: Whether to sort by frequency before saving
            on_load: Optional callback when loading from cache
            on_compute: Optional callback after computing
            verbose: Whether to print default messages (ignored if callbacks provided)

        Returns:
            Counter of word counts
        """
        # Try to load from cache first
        if not force_recompute:
            cached_counts = self.load(source_filename)
            if cached_counts is not None:
                # Execute callback or verbose logging
                if on_load:
                    on_load(cached_counts)
                elif verbose:
                    print(f"✓ Loaded from cache: {len(cached_counts):,} unique words, "
                          f"{sum(cached_counts.values()):,} total occurrences")
                return cached_counts

        # Cache miss or forced recompute
        word_counts = compute_fn()

        # Execute callback or verbose logging
        if on_compute:
            on_compute(word_counts)
        elif verbose:
            print(f"✓ Computed: {len(word_counts):,} unique words, "
                  f"{sum(word_counts.values()):,} total occurrences")

        # Save to cache
        cache_path = self.save(word_counts, source_filename, sort_by_frequency=sort_on_save)
        if verbose and not on_compute:
            print(f"✓ Saved to cache: {cache_path}")

        return word_counts