"""
Domain-specific cache implementation for BPE tokenizer merge operations.

This module provides a specialized cache for storing and retrieving the sequence
of merge operations from BPE training, enabling instant vocabulary reconstruction
without retraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import JSONCache

if TYPE_CHECKING:
    pass


class MergesCache(JSONCache[list[tuple[bytes, bytes]]]):
    """
    Domain-specific cache for BPE tokenizer merge sequence.

    Handles serialization of list[tuple[bytes, bytes]] to JSON format with:
    - Hex-encoded byte pairs for human readability
    - Preserves merge order (critical for BPE algorithm)
    - Cache key includes both filename and vocab_size
    - Integration with training logger for statistics
    """

    def __init__(self, cache_dir: Path | None = None, logger: Any | None = None) -> None:
        """
        Initialize merges cache.

        Args:
            cache_dir: Directory for cache files (default: project_root/data)
            logger: Optional logger with log_stats method (default: None)
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        super().__init__(cache_dir=cache_dir, cache_prefix="merges", logger=logger)

    def _get_default_cache_dir(self) -> Path:
        """Get the default cache directory (project_root/data/cache)."""
        from cs336_basics.utils import get_project_root
        return get_project_root() / "data" / "cache"

    def get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.

        Overrides base class to handle both filenames and full paths gracefully.
        Extracts just the filename if a full path is provided.

        Args:
            key: Cache key in format "{filename}_vocab{vocab_size}"

        Returns:
            Path to the cache file
        """
        # Extract just the filename if a full path was provided (before _vocab suffix)
        if '/' in key or '\\' in key:
            # Split on _vocab to separate path from vocab_size
            parts = key.split('_vocab')
            if len(parts) == 2:
                file_part = Path(parts[0]).name
                key = f"{file_part}_vocab{parts[1]}"
            else:
                key = Path(key).name

        return super().get_cache_path(key)

    def serialize(self, data: list[tuple[bytes, bytes]]) -> dict[str, Any]:
        """
        Serialize merge sequence to JSON format.

        Converts list[tuple[bytes, bytes]] to dictionary with:
        - "merges" key containing array of 2-element hex-encoded byte pairs
        - "count" key with total number of merges (metadata)

        Args:
            data: List of merge operations in order

        Returns:
            JSON-serializable dictionary
        """
        merges_list = []

        for left, right in data:
            # Each merge is a 2-element array of hex-encoded bytes
            merges_list.append([left.hex(), right.hex()])

        return {
            "merges": merges_list,
            "count": len(merges_list)
        }

    def deserialize(self, json_data: dict[str, Any]) -> list[tuple[bytes, bytes]]:
        """
        Deserialize JSON data back to merge sequence.

        Converts dictionary back to list[tuple[bytes, bytes]].

        Args:
            json_data: Dictionary loaded from JSON file

        Returns:
            List of merge operations in original order

        Raises:
            ValueError: If data format is invalid
        """
        if "merges" not in json_data:
            raise ValueError("Invalid merges cache format: missing 'merges' key")

        merges_list: list[tuple[bytes, bytes]] = []

        for merge_pair in json_data["merges"]:
            if not isinstance(merge_pair, list) or len(merge_pair) != 2:
                raise ValueError(f"Expected merge pair with 2 elements, got {merge_pair}")

            left = bytes.fromhex(merge_pair[0])
            right = bytes.fromhex(merge_pair[1])
            merges_list.append((left, right))

        return merges_list

    def _log_load(self, cache_path: Path, data: list[tuple[bytes, bytes]]) -> None:
        """Log statistics when loading merges."""
        if self.logger and hasattr(self.logger, 'log_stats'):
            self.logger.log_stats(
                **{
                    "Loaded merges from cache": "",
                    "Total merges": len(data)
                }
            )

    def make_key(self, filename: str, vocab_size: int) -> str:
        """
        Create cache key from filename and vocab_size.

        Args:
            filename: Source data filename
            vocab_size: Target vocabulary size

        Returns:
            Cache key in format "{filename}_vocab{vocab_size}"
        """
        # Extract just filename if full path provided
        if '/' in filename or '\\' in filename:
            filename = Path(filename).name

        return f"{filename}_vocab{vocab_size}"
