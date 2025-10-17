"""
Universal JSON-based cache system.

This module provides a generic, reusable base class for JSON file caching
that can be extended for any domain-specific caching needs.

Features:
- Atomic file writes for thread safety
- Automatic cache directory management
- Flexible serialization/deserialization hooks
- Error handling with fallback behavior
- Optional logging integration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Generic, Callable, Any

if TYPE_CHECKING:
    pass


T = TypeVar('T')


class JSONCache(Generic[T]):
    """
    Universal JSON-based cache service with customizable serialization.

    This base class provides reusable caching functionality for any data type
    that can be serialized to JSON. Subclasses define how to serialize/deserialize
    their specific data types.

    Features:
    - Atomic file writes (write to temp, then rename)
    - Automatic cache directory management
    - Flexible serialization hooks
    - Error handling with fallback behavior
    - Optional logging integration
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_prefix: str = "cache",
        logger: Any | None = None
    ) -> None:
        """
        Initialize the JSON cache service.

        Args:
            cache_dir: Directory for cache files (default: current directory)
            cache_prefix: Prefix for cache filenames (default: "cache")
            logger: Optional logger object (default: None, silent mode)
        """
        self.cache_dir = cache_dir or Path.cwd()
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_prefix = cache_prefix
        self.logger = logger

    def get_cache_path(self, key: str) -> Path:
        """
        Get the cache file path for a given key.

        Args:
            key: Cache key (typically a filename or identifier)

        Returns:
            Path to the cache file
        """
        cache_filename = f"{self.cache_prefix}_{key}.json"
        return self.cache_dir / cache_filename

    def exists(self, key: str) -> bool:
        """
        Check if cache file exists for the given key.

        Args:
            key: Cache key to check

        Returns:
            True if cache exists, False otherwise
        """
        return self.get_cache_path(key).exists()

    def serialize(self, data: T) -> dict[str, Any]:
        """
        Convert data to JSON-serializable format.

        Subclasses should override this method to define custom serialization.

        Args:
            data: Data to serialize

        Returns:
            JSON-serializable dictionary

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement serialize()")

    def deserialize(self, json_data: dict[str, Any]) -> T:
        """
        Convert JSON data back to original format.

        Subclasses should override this method to define custom deserialization.

        Args:
            json_data: Dictionary loaded from JSON

        Returns:
            Deserialized data in original format

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError("Subclasses must implement deserialize()")

    def save(
        self,
        data: T,
        key: str,
        indent: int = 2,
        sort_keys: bool = False
    ) -> Path:
        """
        Save data to a JSON cache file with atomic write.

        Args:
            data: Data to cache
            key: Cache key (used in filename)
            indent: JSON indentation level (default: 2)
            sort_keys: Whether to sort JSON keys (default: False)

        Returns:
            Path to the saved cache file

        Raises:
            RuntimeError: If save operation fails
        """
        cache_path = self.get_cache_path(key)

        # Serialize data to JSON-compatible format
        serializable_data = self.serialize(data)

        # Atomic write: write to temp file, then rename
        temp_path = cache_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=indent, sort_keys=sort_keys)

            # Atomic rename (POSIX systems guarantee atomicity)
            temp_path.replace(cache_path)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save cache to {cache_path}: {e}") from e

        # Optional logging
        self._log_save(cache_path)

        return cache_path

    def load(self, key: str) -> T | None:
        """
        Load data from a cached JSON file.

        Args:
            key: Cache key to load

        Returns:
            Deserialized data if cache exists and is valid, None otherwise
        """
        cache_path = self.get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Deserialize JSON data to original format
            data = self.deserialize(json_data)

            # Optional logging
            self._log_load(cache_path, data)

            return data

        except (json.JSONDecodeError, ValueError, KeyError, IOError) as e:
            self._log_error(cache_path, e)
            return None

    def load_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        force_recompute: bool = False,
        save_kwargs: dict[str, Any] | None = None
    ) -> T:
        """
        Load from cache or compute and save if not available.

        This is the primary method for working with cached data - it handles
        the entire load-or-compute-and-save workflow.

        Args:
            key: Cache key
            compute_fn: Function to call on cache miss (returns data to cache)
            force_recompute: If True, bypass cache and recompute (default: False)
            save_kwargs: Optional kwargs to pass to save() method

        Returns:
            Cached or computed data
        """
        # Try to load from cache first (unless forced to recompute)
        if not force_recompute:
            cached_data = self.load(key)
            if cached_data is not None:
                return cached_data

        # Cache miss or forced recompute - run computation
        data = compute_fn()

        # Save to cache for future use
        save_kwargs = save_kwargs or {}
        self.save(data, key, **save_kwargs)

        return data

    def _log_save(self, cache_path: Path) -> None:
        """Hook for logging save operations (can be overridden)."""
        if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
            print(f"Saved: {cache_path.name}")

    def _log_load(self, cache_path: Path, data: T) -> None:
        """Hook for logging load operations (can be overridden)."""
        pass  # Override in subclass for custom logging

    def _log_error(self, cache_path: Path, error: Exception) -> None:
        """Hook for logging errors (can be overridden)."""
        print(f"Warning: Failed to load cache from {cache_path}: {error}")