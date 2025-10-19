"""BPE tokenizer with memory-efficient training and deterministic merges.

Implements GPT-2/GPT-5 compatible byte-pair encoding with multiprocessing support.
Training uses chunked file processing (300MB chunks) to handle large datasets without
loading entire files into memory.

Algorithm: Iteratively merges most frequent byte pairs using lexicographic tie-breaking
for deterministic results. Special tokens are kept atomic and never split.

Classes:
    BPETokenizer: Main tokenizer for encoding/decoding with trained vocabularies
    TrainResult: Training output containing vocab and merge operations
"""

import regex as re
from dataclasses import dataclass
from cs336_basics.assignment_1.bpe_trainer import BPETrainer
from cs336_basics.utils import MergesCache, VocabCache
from cs336_basics.constants import (
    GPT5_PAT_STR,
    MODEL_DEFAULTS,
    TINYSTORIES_VALID, TINYSTORIES_TRAIN, OWT_TRAIN, GPT2_PAT_STR
)


@dataclass
class TrainResult:
    """Result of BPE training."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


class BPETokenizer:
    """BPE (Byte Pair Encoding) Tokenizer for CS336 Assignment 1."""

    def __init__(self, pattern: str, special_tokens: list[str], verbose: bool = False) -> None:
        """Initialize BPE tokenizer.

        Args:
            pattern: Regex pattern for pre-tokenization (GPT2_PAT_STR or GPT5_PAT_STR)
            special_tokens: List of special tokens (e.g., ['<|endoftext|>'])
            verbose: Whether to print progress information
        """
        self.pretokenization_pattern = re.compile(pattern)
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: list[str] = special_tokens
        self.verbose = verbose

        # Initialize caches for final results
        self.merges_cache = MergesCache(logger=None)
        self.vocab_cache = VocabCache(logger=None)

        # Build tokenization pattern: special tokens | regular pattern
        if special_tokens:
            special_pattern = '|'.join(re.escape(token) for token in special_tokens)
            self.tokenization_pattern = re.compile(f'({special_pattern})|{pattern}')
        else:
            self.tokenization_pattern = self.pretokenization_pattern

    def init_vocab(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        """Initialize vocabulary: special tokens + base bytes + merge results.

        Vocab IDs: [0..n-1]: special tokens, [n..n+255]: bytes, [n+256..]: merges
        """
        self.vocab = {}

        # First: Special tokens (0, 1, 2, ...)
        for token_idx, token in enumerate(self.special_tokens):
            self.vocab[token_idx] = token.encode('utf-8')

        num_special = len(self.special_tokens)

        # Second: Base bytes (256 individual bytes)
        for byte_value in range(256):
            self.vocab[num_special + byte_value] = bytes([byte_value])

        # Third: Merges (if provided from training)
        if merges:
            for merge_idx, (left, right) in enumerate(merges):
                self.vocab[num_special + 256 + merge_idx] = left + right

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token IDs.

        Args:
            text: Input text string to encode

        Returns:
            List of token IDs from the vocabulary
        """
        # TODO: Implement BPE encoding
        # 1. Pre-tokenize text using self.tokenization_pattern
        # 2. For each token, split into bytes
        # 3. Apply merges in order to get final tokens
        # 4. Map tokens to IDs using vocab
        raise NotImplementedError("encode() method needs implementation")

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string
        """
        # TODO: Implement BPE decoding
        # 1. Map each token ID to bytes using self.vocab
        # 2. Concatenate all bytes
        # 3. Decode bytes to UTF-8 string
        raise NotImplementedError("decode() method needs implementation")

    def train(self, input_path: str, vocab_size: int) -> TrainResult:
        """Train BPE tokenizer on input data.

        Args:
            input_path: Path to training data file
            vocab_size: Target vocabulary size

        Returns:
            TrainResult with vocabulary and merge operations
        """
        # Try to load cached vocab and merges first (biggest speedup!)
        cache_key = self.merges_cache.make_key(input_path, vocab_size)
        cached_merges = self.merges_cache.load(cache_key)
        cached_vocab = self.vocab_cache.load(cache_key)

        if cached_merges is not None and cached_vocab is not None:
            # Both cached - use directly
            self.vocab = cached_vocab
            return TrainResult(vocab=self.vocab, merges=cached_merges)

        if cached_merges is not None:
            # Only merges cached - build vocab
            self.init_vocab(cached_merges)
            self.vocab_cache.save(self.vocab, cache_key)
            return TrainResult(vocab=self.vocab, merges=cached_merges)

        # Cache miss - run full training using BPETrainer
        trainer = BPETrainer(
            special_tokens=self.special_tokens,
            tokenization_pattern=self.tokenization_pattern,
            verbose=self.verbose
        )

        # Initialize training state
        state = trainer.initialize_training_state(input_path, vocab_size)

        # Run merge loop
        merges = trainer.run_merge_loop(state, checkpoint_interval=300)

        # Build vocabulary
        trainer.logger.log_step("Step 3: Building vocabulary...", "vocab")
        self.init_vocab(merges)
        trainer.logger.log_complete("Vocabulary built", "vocab")

        # Save both merges and vocab to cache for future runs
        self.merges_cache.save(merges, cache_key)
        self.vocab_cache.save(self.vocab, cache_key)

        # Log summary
        trainer.logger.log_training_summary("train", "word_counting", "merges", "vocab")

        return TrainResult(vocab=self.vocab, merges=merges)


# TODO: Remove this debug code when implementing full tokenizer
# This is currently here for testing the basic functionality
if __name__ == "__main__":
    bpe = BPETokenizer(GPT2_PAT_STR, MODEL_DEFAULTS.DEFAULT_SPECIAL_TOKENS, verbose=True)
    train_result = bpe.train(TINYSTORIES_TRAIN, 10_000)
    print('vocab size = ', len(train_result.vocab))
    print('merges count = ', len(train_result.merges))
    print('First 30 longest merges:', sorted(train_result.merges, key=lambda x: len(x[0] + x[1]), reverse=True)[:30])

