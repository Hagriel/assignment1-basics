"""
Domain-specific cache implementation for BPE tokenizer vocabulary.

This module provides a specialized cache for storing and retrieving vocabulary
mappings from BPE training, enabling instant tokenizer initialization without
reconstructing vocab from merges.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import JSONCache, get_default_cache_dir, encode_bytes_for_json, decode_bytes_from_json

if TYPE_CHECKING:
    pass


class VocabCache(JSONCache[dict[int, bytes]]):
    """
    Domain-specific cache for BPE tokenizer vocabulary.

    Handles serialization of dict[int, bytes] to JSON format with:
    - Hex-encoded byte values for human readability
    - Cache key includes both filename and vocab_size
    - Integration with training logger for statistics
    """

    def __init__(self, cache_dir: Path | None = None, logger: Any | None = None) -> None:
        """
        Initialize vocab cache.

        Args:
            cache_dir: Directory for cache files (default: project_root/data)
            logger: Optional logger with log_stats method (default: None)
        """
        if cache_dir is None:
            cache_dir = get_default_cache_dir()

        super().__init__(cache_dir=cache_dir, cache_prefix="vocab", logger=logger)

    def get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.

        Organizes cache files by dataset name in subdirectories.
        Creates: data/cache/{dataset_name}/vocab_vocab{size}.json

        Args:
            key: Cache key in format "{filename}_vocab{vocab_size}"

        Returns:
            Path to the cache file
        """
        # Extract the filename part (before _vocab suffix)
        if '/' in key or '\\' in key:
            # Split on _vocab to separate path from vocab_size
            parts = key.split('_vocab')
            if len(parts) == 2:
                file_part = Path(parts[0]).name
                key = f"{file_part}_vocab{parts[1]}"
            else:
                key = Path(key).name

        # Get dataset name (part before _vocab)
        if '_vocab' in key:
            dataset_name = key.split('_vocab')[0]
        else:
            dataset_name = key

        # Create subdirectory for this dataset
        dataset_dir = self.cache_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Include vocab_size in filename: vocab_vocab500.json
        if '_vocab' in key:
            vocab_suffix = key.split('_vocab')[1]
            return dataset_dir / f"{self.cache_prefix}_vocab{vocab_suffix}.json"
        else:
            return dataset_dir / f"{self.cache_prefix}.json"

    def serialize(self, data: dict[int, bytes]) -> dict[str, Any]:
        """
        Serialize vocab to JSON format.

        Converts dict[int, bytes] to dictionary with:
        - Keys: string representation of token IDs
        - Values: byte strings (UTF-8 if possible, hex prefixed with "\\x" otherwise)

        Args:
            data: Vocabulary mapping token IDs to bytes

        Returns:
            JSON-serializable dictionary
        """
        serializable_data = {}

        for token_id, token_bytes in data.items():
            serializable_data[str(token_id)] = encode_bytes_for_json(token_bytes)

        return serializable_data

    def deserialize(self, json_data: dict[str, Any]) -> dict[int, bytes]:
        """
        Deserialize JSON data back to vocab.

        Converts dictionary back to dict[int, bytes].

        Args:
            json_data: Dictionary loaded from JSON file

        Returns:
            Vocabulary mapping token IDs to bytes

        Raises:
            ValueError: If data format is invalid
        """
        vocab: dict[int, bytes] = {}

        for key, value in json_data.items():
            token_id = int(key)
            vocab[token_id] = decode_bytes_from_json(value)

        return vocab

    def _log_load(self, cache_path: Path, data: dict[int, bytes]) -> None:
        """Log statistics when loading vocab."""
        if self.logger and hasattr(self.logger, 'log_stats'):
            self.logger.log_stats(
                **{
                    "Loaded vocab from cache": "",
                    "Vocab size": len(data)
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
