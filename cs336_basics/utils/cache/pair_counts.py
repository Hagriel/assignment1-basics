"""
Domain-specific cache implementation for BPE tokenizer pair counts.

This module provides a specialized cache for storing and retrieving byte pair frequency
counts used in BPE tokenizer training, with support for hex-encoded byte pairs
and frequency-based sorting.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING, Any

from .base import JSONCache

if TYPE_CHECKING:
    pass


class PairCountsCache(JSONCache[Counter[tuple[bytes, bytes]]]):
    """
    Domain-specific cache for BPE tokenizer byte pair frequency counts.

    Handles serialization of Counter[tuple[bytes, bytes]] to JSON format with:
    - Hex-encoded byte pairs for human readability
    - Optional sorting by frequency with lexicographic tie-breaking
    - Integration with training logger for statistics
    """

    def __init__(self, cache_dir: Path | None = None, logger: Any | None = None) -> None:
        """
        Initialize pair counts cache.

        Args:
            cache_dir: Directory for cache files (default: project_root/data)
            logger: Optional logger with log_stats method (default: None)
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        super().__init__(cache_dir=cache_dir, cache_prefix="pair_counts", logger=logger)
        self._sort_by_frequency = True  # Default behavior for pair counts

    def _get_default_cache_dir(self) -> Path:
        """Get the default cache directory (project_root/data/cache)."""
        from cs336_basics.utils import get_project_root
        return get_project_root() / "data" / "cache"

    def get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.

        Organizes cache files by dataset name in subdirectories.
        Creates: data/cache/{dataset_name}/pair_counts.json

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

    def serialize(self, data: Counter[tuple[bytes, bytes]]) -> dict[str, Any]:
        """
        Serialize pair counts to JSON format.

        Converts Counter[tuple[bytes, bytes]] to dictionary with:
        - Keys: JSON arrays of 2 byte strings (UTF-8 if possible, hex prefixed with "\\x" otherwise)
        - Values: frequency counts
        - Optionally sorted by frequency (descending) with lexicographic tie-breaking

        Args:
            data: Counter mapping byte pairs to their frequencies

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

        # Convert each pair to string array (UTF-8 or hex)
        for (left, right), count in items:
            key_parts = []
            for b in [left, right]:
                try:
                    # Try to decode as UTF-8
                    decoded = b.decode('utf-8')
                    key_parts.append(decoded)
                except UnicodeDecodeError:
                    # Fall back to hex with \x prefix
                    key_parts.append(f"\\x{b.hex()}")

            key = json.dumps(key_parts)
            serializable_data[key] = count

        return serializable_data

    def deserialize(self, json_data: dict[str, Any]) -> Counter[tuple[bytes, bytes]]:
        """
        Deserialize JSON data back to pair counts.

        Converts dictionary with UTF-8/hex keys back to Counter[tuple[bytes, bytes]].

        Args:
            json_data: Dictionary loaded from JSON file

        Returns:
            Counter mapping byte pairs to frequencies

        Raises:
            ValueError: If data format is invalid
        """
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for key, count in json_data.items():
            # Parse the JSON array
            str_list = json.loads(key)
            if len(str_list) != 2:
                raise ValueError(f"Expected pair with 2 elements, got {len(str_list)}")

            pair_parts = []
            for s in str_list:
                if isinstance(s, str) and s.startswith("\\x"):
                    # Hex-encoded bytes
                    hex_str = s[2:]  # Remove \x prefix
                    pair_parts.append(bytes.fromhex(hex_str))
                else:
                    # UTF-8 string: encode back to bytes
                    pair_parts.append(s.encode('utf-8'))

            pair_counts[(pair_parts[0], pair_parts[1])] = count

        return pair_counts

    def save(
        self,
        data: Counter[tuple[bytes, bytes]],
        key: str,
        sort_by_frequency: bool = False,
        **kwargs: Any
    ) -> Path:
        """
        Save pair counts to cache with frequency sorting option.

        Args:
            data: Pair counts to save
            key: Cache key (typically source filename)
            sort_by_frequency: Whether to sort by frequency before saving
            **kwargs: Additional arguments passed to parent save()

        Returns:
            Path to saved cache file
        """
        # Set sorting preference before serialization
        self._sort_by_frequency = sort_by_frequency

        return super().save(data, key, **kwargs)

    def _log_load(self, cache_path: Path, data: Counter[tuple[bytes, bytes]]) -> None:
        """Log statistics when loading pair counts."""
        if self.logger and hasattr(self.logger, 'log_stats'):
            self.logger.log_stats(
                **{
                    "Loaded pair counts from cache": "",
                    "Total unique pairs": len(data),
                    "Total pair occurrences": sum(data.values())
                }
            )
