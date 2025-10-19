"""Data management for BPE tokenizer.

Handles vocabulary loading, caching, and state management.
Separates data I/O concerns from core tokenization logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cs336_basics.utils import MergesCache, VocabCache

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class VocabState:
    """Immutable vocabulary state.

    Attributes:
        vocab: Mapping from token IDs to byte sequences
        merges: Ordered list of byte pair merges
        bytes_to_id: Reverse mapping for efficient encoding
    """
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    bytes_to_id: dict[bytes, int]


class TokenizerDataManager:
    """Manages vocabulary loading, caching, and state for BPE tokenizer.

    Responsibilities:
        - Load vocabulary from cache or direct data
        - Build vocabulary from merges
        - Maintain reverse mappings for encoding
        - Handle cache key generation
        - Validate input parameters
    """

    __slots__ = ('merges_cache', 'vocab_cache', 'verbose')

    def __init__(self, verbose: bool = False) -> None:
        """Initialize data manager.

        Args:
            verbose: Whether to print progress information
        """
        self.merges_cache = MergesCache(logger=None)
        self.vocab_cache = VocabCache(logger=None)
        self.verbose = verbose

    def build_vocab(
        self,
        special_tokens: list[str],
        merges: list[tuple[bytes, bytes]] | None = None
    ) -> dict[int, bytes]:
        """Build vocabulary from special tokens, base bytes, and merges.

        Vocab structure:
            IDs [0..n-1]: special tokens
            IDs [n..n+255]: individual bytes (0x00-0xFF)
            IDs [n+256..]: merged tokens from BPE training

        Args:
            special_tokens: List of special tokens
            merges: Optional list of merge operations

        Returns:
            Complete vocabulary mapping
        """
        vocab = {}

        # Special tokens (IDs 0, 1, 2, ...)
        for token_idx, token in enumerate(special_tokens):
            vocab[token_idx] = token.encode('utf-8')

        num_special = len(special_tokens)

        # Base bytes (256 individual bytes)
        for byte_value in range(256):
            vocab[num_special + byte_value] = bytes([byte_value])

        # Merged tokens from BPE
        if merges:
            for merge_idx, (left, right) in enumerate(merges):
                vocab[num_special + 256 + merge_idx] = left + right

        return vocab

    def build_reverse_mapping(self, vocab: dict[int, bytes]) -> dict[bytes, int]:
        """Build reverse vocabulary mapping for O(1) encoding lookups.

        Args:
            vocab: Forward vocabulary mapping

        Returns:
            Reverse mapping from bytes to token IDs
        """
        return {token_bytes: token_id for token_id, token_bytes in vocab.items()}

    def create_vocab_state(
        self,
        special_tokens: list[str],
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None
    ) -> VocabState:
        """Create complete vocabulary state with reverse mapping.

        Args:
            special_tokens: List of special tokens
            vocab: Pre-built vocabulary (if None, built from merges)
            merges: List of merge operations (required if vocab is None)

        Returns:
            Immutable vocabulary state

        Raises:
            ValueError: If neither vocab nor merges provided
        """
        if vocab is not None and merges is not None:
            # Both provided - use as-is
            bytes_to_id = self.build_reverse_mapping(vocab)
            return VocabState(vocab=vocab, merges=merges, bytes_to_id=bytes_to_id)

        if merges is not None:
            # Build vocab from merges
            vocab = self.build_vocab(special_tokens, merges)
            bytes_to_id = self.build_reverse_mapping(vocab)
            return VocabState(vocab=vocab, merges=merges, bytes_to_id=bytes_to_id)

        raise ValueError("Must provide either vocab or merges")

    def load_from_cache(
        self,
        dataset_name: str,
        vocab_size: int
    ) -> tuple[dict[int, bytes] | None, list[tuple[bytes, bytes]] | None]:
        """Load vocabulary and merges from cache.

        Args:
            dataset_name: Name of dataset used for training
            vocab_size: Target vocabulary size

        Returns:
            Tuple of (vocab, merges), with None values if not cached
        """
        cache_key = self.merges_cache.make_key(dataset_name, vocab_size)
        cached_vocab = self.vocab_cache.load(cache_key)
        cached_merges = self.merges_cache.load(cache_key)
        return cached_vocab, cached_merges

    def save_to_cache(
        self,
        dataset_name: str,
        vocab_size: int,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]]
    ) -> None:
        """Save vocabulary and merges to cache.

        Args:
            dataset_name: Name of dataset used for training
            vocab_size: Target vocabulary size
            vocab: Vocabulary mapping to cache
            merges: Merge operations to cache
        """
        cache_key = self.merges_cache.make_key(dataset_name, vocab_size)
        self.vocab_cache.save(vocab, cache_key)
        self.merges_cache.save(merges, cache_key)

    def load_vocab_from_direct(
        self,
        special_tokens: list[str],
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None
    ) -> VocabState | None:
        """Load vocabulary from direct data.

        Args:
            special_tokens: List of special tokens
            vocab: Pre-built vocabulary mapping
            merges: List of merge operations

        Returns:
            VocabState if data provided, None otherwise

        Raises:
            ValueError: If invalid combination of arguments
        """
        if vocab is None and merges is None:
            return None

        if merges is None:
            raise ValueError("Must provide at least merges (or both vocab and merges)")

        return self.create_vocab_state(special_tokens, vocab, merges)

    def load_vocab_from_cache(
        self,
        special_tokens: list[str],
        dataset_name: str,
        vocab_size: int
    ) -> VocabState | None:
        """Load vocabulary from cache.

        Args:
            special_tokens: List of special tokens
            dataset_name: Name of dataset for cache lookup
            vocab_size: Vocabulary size for cache lookup

        Returns:
            VocabState if found in cache, None otherwise
        """
        cached_vocab, cached_merges = self.load_from_cache(dataset_name, vocab_size)

        if cached_vocab is not None and cached_merges is not None:
            # Both cached
            if self.verbose:
                print(f"Loaded vocabulary from cache: {dataset_name}, vocab_size={vocab_size}")
            return self.create_vocab_state(special_tokens, cached_vocab, cached_merges)

        if cached_merges is not None:
            # Only merges cached - rebuild vocab and save
            state = self.create_vocab_state(special_tokens, None, cached_merges)
            cache_key = self.merges_cache.make_key(dataset_name, vocab_size)
            self.vocab_cache.save(state.vocab, cache_key)
            if self.verbose:
                print(f"Built vocabulary from cached merges: {dataset_name}, vocab_size={vocab_size}")
            return state

        if self.verbose:
            print(f"No cached data found: {dataset_name}, vocab_size={vocab_size}")
        return None

    def load_vocab(
        self,
        special_tokens: list[str],
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        dataset_name: str | None = None,
        vocab_size: int | None = None
    ) -> VocabState:
        """Load vocabulary from either direct data or cache.

        Use either (vocab, merges) OR (dataset_name, vocab_size), not both.

        Args:
            special_tokens: List of special tokens
            vocab: Pre-built vocabulary mapping
            merges: List of merge operations
            dataset_name: Dataset name for cache lookup
            vocab_size: Vocabulary size for cache lookup

        Returns:
            VocabState with loaded vocabulary

        Raises:
            ValueError: If invalid argument combination or cache miss

        Examples:
            >>> manager.load_vocab(special_tokens, vocab=my_vocab, merges=my_merges)
            >>> manager.load_vocab(special_tokens, dataset_name="TinyStories", vocab_size=8192)
        """
        direct_data = vocab is not None or merges is not None
        cache_data = dataset_name is not None or vocab_size is not None

        if direct_data and cache_data:
            raise ValueError("Provide either (vocab, merges) OR (dataset_name, vocab_size), not both")

        if not direct_data and not cache_data:
            raise ValueError("Must provide either (vocab, merges) OR (dataset_name, vocab_size)")

        # Load from direct data
        if direct_data:
            state = self.load_vocab_from_direct(special_tokens, vocab, merges)
            if state is None:
                raise ValueError("Must provide at least merges")
            return state

        # Load from cache
        if dataset_name is None or vocab_size is None:
            raise ValueError("Both dataset_name and vocab_size must be provided for cache lookup")
        state = self.load_vocab_from_cache(special_tokens, dataset_name, vocab_size)
        if state is None:
            raise ValueError(f"Cache miss: {dataset_name}, vocab_size={vocab_size}")
        return state

    def try_load_vocab(
        self,
        special_tokens: list[str],
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        dataset_name: str | None = None,
        vocab_size: int | None = None
    ) -> VocabState | None:
        """Try to load vocabulary, returning None on cache miss instead of raising.

        Args:
            special_tokens: List of special tokens
            vocab: Pre-built vocabulary mapping
            merges: List of merge operations
            dataset_name: Dataset name for cache lookup
            vocab_size: Vocabulary size for cache lookup

        Returns:
            VocabState if successful, None if cache miss

        Raises:
            ValueError: If invalid argument combination
        """
        direct_data = vocab is not None or merges is not None
        cache_data = dataset_name is not None or vocab_size is not None

        if direct_data and cache_data:
            raise ValueError("Provide either (vocab, merges) OR (dataset_name, vocab_size), not both")

        if not direct_data and not cache_data:
            raise ValueError("Must provide either (vocab, merges) OR (dataset_name, vocab_size)")

        # Load from direct data
        if direct_data:
            return self.load_vocab_from_direct(special_tokens, vocab, merges)

        # Load from cache (returns None on miss)
        if dataset_name is None or vocab_size is None:
            raise ValueError("Both dataset_name and vocab_size must be provided for cache lookup")
        return self.load_vocab_from_cache(special_tokens, dataset_name, vocab_size)