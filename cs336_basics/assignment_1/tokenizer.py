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

        escaped_special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        self.special_tokens_pattern = re.compile(escaped_special_pattern)
        self.special_tokens_split_pattern = re.compile(f"({escaped_special_pattern})")

        self._bytes_to_id: dict[bytes, int] = {}
        self.data_manager = TokenizerDataManager(verbose=verbose)

    def init_vocab(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        """Initialize vocabulary: special tokens + base bytes + merge results.

        Vocab IDs: [0..n-1]: special tokens, [n..n+255]: bytes, [n+256..]: merges

        Args:
            merges: Optional list of merge operations to build vocab
        """
        self.vocab = self.data_manager.build_vocab(self.special_tokens, merges)
        self._build_reverse_vocab()

    def _build_reverse_vocab(self) -> None:
        """Build reverse vocabulary mapping for O(1) encoding lookups."""
        self._bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    def _ensure_vocab_initialized(self) -> None:
        """Validate vocabulary initialization and build reverse mapping if needed.

        Raises:
            ValueError: If vocabulary is not initialized
        """
        if not self.vocab:
            raise ValueError("Vocabulary not initialized. Call init_vocab() or load_vocab() first.")

        if not self._bytes_to_id:
            self._build_reverse_vocab()

    def load_vocab(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        dataset_name: str | None = None,
        vocab_size: int | None = None
    ) -> bool:
        """Load vocabulary either from direct data or from cache.

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

    def _apply_bpe_merges(self, byte_tokens: list[bytes]) -> list[bytes]:
        """Apply learned BPE merge operations to byte token sequence.

        Args:
            byte_tokens: List of byte tokens to merge

        Returns:
            List of merged byte tokens after applying all learned merge rules
        """
        if len(byte_tokens) <= 1:
            return byte_tokens

        for left_token, right_token in self.merges:
            if len(byte_tokens) <= 1:
                break

            merged_tokens = []
            token_position = 0

            while token_position < len(byte_tokens):
                if (token_position < len(byte_tokens) - 1 and
                    byte_tokens[token_position] == left_token and
                    byte_tokens[token_position + 1] == right_token):
                    merged_tokens.append(left_token + right_token)
                    token_position += 2
                else:
                    merged_tokens.append(byte_tokens[token_position])
                    token_position += 1

            byte_tokens = merged_tokens

        return byte_tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token IDs.

        Args:
            text: Input text string to encode

        Returns:
            List of token IDs from the vocabulary
        """
        self._ensure_vocab_initialized()

        token_ids = []
        text_segments = self.special_tokens_split_pattern.split(text)

        for text_segment in text_segments:
            if not text_segment:
                continue

            if text_segment in self.special_tokens:
                special_token_bytes = text_segment.encode('utf-8')
                if special_token_bytes not in self._bytes_to_id:
                    raise ValueError(f"Special token {text_segment!r} not found in vocabulary")
                token_ids.append(self._bytes_to_id[special_token_bytes])
                continue

            for match in self.pretokenization_pattern.finditer(text_segment):
                token_text = match.group()
                byte_tokens = [bytes([byte_value]) for byte_value in token_text.encode('utf-8')]
                byte_tokens = self._apply_bpe_merges(byte_tokens)

                for token_bytes in byte_tokens:
                    if token_bytes not in self._bytes_to_id:
                        raise ValueError(f"Token {token_bytes!r} not found in vocabulary")
                    token_ids.append(self._bytes_to_id[token_bytes])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        self._ensure_vocab_initialized()

        token_bytes_sequence = []
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
            token_bytes_sequence.append(self.vocab[token_id])

        concatenated_bytes = b''.join(token_bytes_sequence)

        try:
            return concatenated_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode bytes to UTF-8: {e}") from e