"""Base tokenizer for encoding and decoding text.

Provides core tokenization functionality including:
- Vocabulary management
- Encoding text to token IDs
- Decoding token IDs to text
- BPE merge application

This base class is extended by BPETokenizer for training functionality.
"""

from __future__ import annotations

import regex as re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regex import Pattern

from cs336_basics.assignment_1.tokenizer_data_manager import TokenizerDataManager


class Tokenizer:
    """Base tokenizer for encoding and decoding text with BPE."""

    def __init__(self, pattern: str, special_tokens: list[str], verbose: bool = False) -> None:
        """Initialize tokenizer.

        Args:
            pattern: Regex pattern for pre-tokenization (GPT2_PAT_STR or GPT5_PAT_STR)
            special_tokens: List of special tokens (e.g., ['<|endoftext|>'])
            verbose: Whether to print progress information
        """
        self.pretokenization_pattern: Pattern = re.compile(pattern)
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.special_tokens: list[str] = special_tokens
        self.verbose: bool = verbose
        # Escape special tokens so | is treated as literal, not alternation operator
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        self.special_tokens_pattern = re.compile(special_pattern)
        # Capturing group so split() includes special tokens in result
        self.tokenization_special_tokens_pattern = re.compile(f"({special_pattern})")

        # Reverse vocab mapping for efficient encoding (cached)
        self._bytes_to_id: dict[bytes, int] = {}

        # Initialize data manager for vocab loading and caching
        self.data_manager = TokenizerDataManager(verbose=verbose)

    def init_vocab(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        """Initialize vocabulary: special tokens + base bytes + merge results.

        Vocab IDs: [0..n-1]: special tokens, [n..n+255]: bytes, [n+256..]: merges

        Args:
            merges: Optional list of merge operations to build vocab
        """
        # Delegate to data manager for vocab building
        self.vocab = self.data_manager.build_vocab(self.special_tokens, merges)

        # Build reverse mapping for encoding
        self._build_reverse_vocab()

    def _build_reverse_vocab(self) -> None:
        """Build reverse vocabulary mapping (bytes -> token_id) for efficient encoding."""
        self._bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    def _ensure_vocab_initialized(self) -> None:
        """Validate that vocabulary has been initialized.

        Raises:
            ValueError: If vocabulary is not initialized
        """
        if not self.vocab:
            raise ValueError("Vocabulary not initialized. Call init_vocab() or load_vocab() first.")

        # Ensure reverse mapping is built for encoding
        if not self._bytes_to_id:
            self._build_reverse_vocab()

    def load_vocab(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        dataset_name: str | None = None,
        vocab_size: int | None = None
    ) -> bool:
        """
        Load vocabulary either from direct data or from cache.

        Use either (vocab, merges) OR (dataset_name, vocab_size), not both.

        Args:
            vocab: Pre-built vocabulary mapping (token_id -> bytes)
            merges: List of merge operations
            dataset_name: Name of dataset to load from cache (e.g., "TinyStoriesV2-GPT4-train")
            vocab_size: Vocabulary size to load from cache

        Returns:
            True if vocab was loaded successfully, False otherwise
        """
        vocab_state = self.data_manager.try_load_vocab(
            special_tokens=self.special_tokens,
            vocab=vocab,
            merges=merges,
            dataset_name=dataset_name,
            vocab_size=vocab_size
        )

        if vocab_state is None:
            return False

        self.vocab = vocab_state.vocab
        self.merges = vocab_state.merges
        self._bytes_to_id = vocab_state.bytes_to_id

        return True

    def _apply_bpe_merges(self, word: list[bytes]) -> list[bytes]:
        """
        Apply BPE merges to a word (list of byte tokens).

        Args:
            word: List of byte tokens to merge

        Returns:
            List of merged byte tokens
        """
        if len(word) <= 1:
            return word

        # Apply merges in the order they were learned
        for merge_pair in self.merges:
            if len(word) <= 1:
                break

            # Find all positions where this pair occurs
            new_word = []
            position = 0

            while position < len(word):
                # Check if we can merge at this position
                if (position < len(word) - 1 and
                    word[position] == merge_pair[0] and
                    word[position + 1] == merge_pair[1]):
                    # Merge the pair
                    new_word.append(merge_pair[0] + merge_pair[1])
                    position += 2
                else:
                    new_word.append(word[position])
                    position += 1

            word = new_word

        return word

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token IDs.

        Args:
            text: Input text string to encode

        Returns:
            List of token IDs from the vocabulary
        """
        self._ensure_vocab_initialized()

        token_ids = []

        # Step 1: Split text on special tokens to handle them separately
        # This prevents the pretokenization pattern from splitting special tokens
        text_chunks = self.tokenization_special_tokens_pattern.split(text)

        for chunk in text_chunks:
            if not chunk:
                continue

            # Check if this chunk is a special token
            if chunk in self.special_tokens:
                # Special tokens map directly to their token ID
                special_token_bytes = chunk.encode('utf-8')
                if special_token_bytes not in self._bytes_to_id:
                    raise ValueError(f"Special token {chunk!r} not found in vocabulary.")
                token_ids.append(self._bytes_to_id[special_token_bytes])
                continue

            # Step 2: Apply pretokenization pattern to regular text
            for match in self.pretokenization_pattern.finditer(chunk):
                matched_text = match.group()

                # Step 3: Convert matched text to individual bytes
                word = [bytes([b]) for b in matched_text.encode('utf-8')]

                # Step 4: Apply BPE merges
                word = self._apply_bpe_merges(word)

                # Step 5: Map final tokens to their IDs
                for token_bytes in word:
                    if token_bytes in self._bytes_to_id:
                        token_ids.append(self._bytes_to_id[token_bytes])
                    else:
                        raise ValueError(f"Token {token_bytes} not found in vocabulary")

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        self._ensure_vocab_initialized()

        # Step 1: Map each token ID to bytes
        byte_sequence = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_sequence.append(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")

        # Step 2: Concatenate all bytes
        combined_bytes = b''.join(byte_sequence)

        # Step 3: Decode bytes to UTF-8 string
        try:
            return combined_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode bytes to UTF-8: {e}") from e