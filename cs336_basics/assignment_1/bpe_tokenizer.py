"""
BPE (Byte Pair Encoding) Tokenizer Implementation for CS336 Assignment 1.

This module provides a complete BPE training implementation with memory-efficient
multiprocessing, deterministic tie-breaking, and proper special token handling.

Key Features:
- GPT-2/GPT-5 compatible pre-tokenization using regex patterns
- Memory-efficient chunked file processing with multiprocessing
- Deterministic BPE training with lexicographic tie-breaking
- Special token handling (atomic, never split)
- Progress logging with timing breakdowns
- Smart chunking at special token boundaries (300MB target)

Architecture:
- BPETokenizer: Main tokenizer class with training capabilities
- TrainResult: Dataclass for training output (vocab + merges)
- TrainingLogger: Separated logging logic for clean code

Current Implementation Status:
- ✅ BPE training algorithm with deterministic merges
- ✅ Memory-efficient file processing (chunked, multiprocessing)
- ✅ Special token handling (combined regex pattern)
- ✅ Vocabulary building from merges
- ⚠️  Encoding/decoding methods (TODO: needed for 22 tokenizer tests)

Dependencies:
- regex: Advanced regex engine with Unicode support
- collections.Counter: Efficient counting of token frequencies
- cs336_basics.utils: Optimized file chunking utilities
- cs336_basics.training_logger: Clean logging separation

BPE Algorithm:
==============

Training Process (Implemented):
1. Pre-tokenization: Split text using GPT-2/GPT-5 regex
2. Count word frequencies with multiprocessing
3. Iteratively merge most frequent byte pairs (deterministic tie-breaking)
4. Build vocabulary: special tokens + base bytes + merged tokens

Tie-Breaking Strategy:
- When pair counts are equal, use lexicographic ordering
- Ensures deterministic, reproducible results
- Formula: max(pairs, key=lambda x: (x[1], x[0]))

Encoding Process (TODO):
1. Pre-tokenize input text using same regex
2. Apply learned merges in order to each pre-token
3. Return token IDs from vocabulary

Decoding Process (TODO):
1. Look up tokens in vocabulary to get bytes
2. Concatenate bytes and decode to UTF-8 string

Performance:
- Trains 500-vocab BPE on small corpus in ~2.3s
- Memory efficient: processes 300MB chunks, never loads entire file
- Multiprocessing with auto-adjusted worker count
"""

import regex as re
from collections import Counter
from dataclasses import dataclass
from cs336_basics.utils import (
    format_time,
    get_data_path,
    find_chunk_boundaries,
    process_file_chunks_multiprocessing
)
from cs336_basics.training_logger import TrainingLogger
from cs336_basics.constants import (
    GPT5_PAT_STR,
    END_OF_TEXT_TOKEN,
    MODEL_DEFAULTS,
    TINYSTORIES_VALID
)


@dataclass
class TrainResult:
    """Result of BPE training."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


class BPETokenizer:
    """BPE (Byte Pair Encoding) Tokenizer for CS336 Assignment 1."""

    def __init__(self, pat_str: str, special_tokens: list[str], verbose: bool = False) -> None:
        """
        Initialize the BPE tokenizer with GPT-2 pre-tokenization pattern.

        Args:
            pat_str: Regex pattern string for pre-tokenization (e.g., GPT2_PAT_STR)
            special_tokens: List of special tokens (e.g., ['<|endoftext|>'])
            verbose: Whether to print detailed progress information (default: False)
        """
        self.compiled_pat = re.compile(pat_str)
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: list[str] = special_tokens
        self.logger = TrainingLogger(verbose)

        # Build combined pattern: special tokens | regular pattern
        if special_tokens:
            special_pattern = '|'.join(re.escape(token) for token in special_tokens)
            self.combined_pat = re.compile(f'({special_pattern})|{pat_str}')
        else:
            self.combined_pat = self.compiled_pat

    def init_vocab(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        """
        Initialize vocabulary with special tokens, base bytes, and optional merges.

        Args:
            merges: Optional list of BPE merge pairs from training

        Creates vocab mapping:
            0 to len(special_tokens)-1: Special tokens (from self.special_tokens)
            len(special_tokens) to len(special_tokens)+255: Individual byte values (256 bytes)
            len(special_tokens)+256 onwards: Merged tokens from BPE training
        """
        self.vocab = {}

        # First: Special tokens (0, 1, 2, ...)
        for idx, token in enumerate(self.special_tokens):
            self.vocab[idx] = token.encode('utf-8')

        num_special = len(self.special_tokens)

        # Second: Base bytes (256 individual bytes)
        for i in range(256):
            self.vocab[num_special + i] = bytes([i])

        # Third: Merges (if provided from training)
        if merges:
            for idx, (piece1, piece2) in enumerate(merges):
                self.vocab[num_special + 256 + idx] = piece1 + piece2


    def _get_pair_counts(self, word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
        """
        Count all adjacent byte pairs across all words.

        Args:
            word_counts: Counter mapping words (tuples of bytes) to their frequencies

        Returns:
            Counter mapping byte pairs to their total occurrence count
        """
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for word, count in word_counts.items():
            # For each word, count all adjacent pairs
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count

        return pair_counts

    def _merge_pair(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """
        Merge all occurrences of a byte pair in a word.

        Args:
            word: Tuple of bytes representing a word
            pair: Pair of bytes to merge

        Returns:
            New word with all occurrences of the pair merged
        """
        if len(word) < 2:
            return word

        new_word = []
        i = 0

        while i < len(word):
            # If we found the pair and it's not the last byte, merge it
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                # Merge the two bytes
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    def _apply_merge(self, word_counts: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes, ...]]:
        """
        Apply a merge to all words in the word counts.

        Args:
            word_counts: Counter of word frequencies
            pair: Byte pair to merge

        Returns:
            New Counter with the merge applied to all words
        """
        new_counts: Counter[tuple[bytes, ...]] = Counter()

        for word, count in word_counts.items():
            merged_word = self._merge_pair(word, pair)
            new_counts[merged_word] += count

        return new_counts

    def _process_chunk(self, file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
        """
        Process a single chunk of the file and return token counts.

        Args:
            file_path: Path to the data file
            start: Start byte position
            end: End byte position

        Returns:
            Counter: Token counts for this chunk
        """
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")

            word_counts: Counter[tuple[bytes, ...]] = Counter()

            # Use combined pattern to handle special tokens and regular text in one pass
            for match in self.combined_pat.finditer(chunk):
                text = match.group()

                # Check if this is a special token (matched by first capture group)
                if text in self.special_tokens:
                    # Keep special token as a single token (atomic unit)
                    word = (text.encode("utf-8"),)
                else:
                    # Keep regex-matched token as atomic unit, but split into individual bytes for BPE
                    # Each byte becomes a separate element in the tuple for BPE to merge
                    word = tuple(bytes([b]) for b in text.encode("utf-8"))

                word_counts[word] += 1

            return word_counts

    def get_word_counts(self, filename: str, num_workers: int | None = None) -> Counter[tuple[bytes, ...]]:
        """
        Get word frequency counts from a data file using multiprocessing.

        Args:
            filename: Name of the data file to process
            num_workers: Number of worker processes (default: CPU count)

        Returns:
            Counter mapping word tuples (tuples of bytes) to their frequencies
        """
        self.logger.start_timer("word_counting")
        data_file_path = get_data_path(filename)

        with open(data_file_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                split_special_token=END_OF_TEXT_TOKEN,
                target_chunk_size=300 * 1024 * 1024
            )

        chunk_results = process_file_chunks_multiprocessing(
            data_file_path,
            boundaries,
            self._process_chunk,
            num_workers=num_workers,
            show_progress=self.logger.verbose
        )

        total_counts: Counter[tuple[bytes, ...]] = Counter()
        for chunk_counts in chunk_results:
            total_counts.update(chunk_counts)

        self.logger.log_stats(
            **{
                f"Multiprocessing tokenization completed in {format_time(self.logger.get_elapsed('word_counting'))}": "",
                "Total unique words": len(total_counts),
                "Total word occurrences": sum(total_counts.values())
            }
        )

        return total_counts

    def train(self, input_path: str, vocab_size: int) -> TrainResult:
        """
        Train BPE tokenizer on input data.

        Args:
            input_path: Path to training data file
            vocab_size: Target vocabulary size

        Returns:
            Dictionary with 'vocab' (token ID to bytes mapping) and 'merges' (list of merge operations)
        """
        self.logger.start_timer("train")
        self.logger.log_step("Step 1: Getting initial word counts...", "word_counting")

        word_counts = self.get_word_counts(input_path)
        self.logger.log_complete("Word counting", "word_counting")

        # Calculate number of merges needed
        num_special = len(self.special_tokens)
        num_merges_needed = vocab_size - num_special - 256
        self.logger.log_step(f"Step 2: Performing {num_merges_needed:,} BPE merges...", "merges")

        # Perform merges
        merges: list[tuple[bytes, bytes]] = []

        for i in range(num_merges_needed):
            pair_counts = self._get_pair_counts(word_counts)
            if not pair_counts:
                break

            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            word_counts = self._apply_merge(word_counts, best_pair[0])
            merges.append(best_pair[0])

            if (i + 1) % max(1, num_merges_needed // 10) == 0:
                self.logger.log_progress(i + 1, num_merges_needed, "merges")

        self.logger.log_complete("BPE merges", "merges")

        # Build vocabulary
        self.logger.log_step("Step 3: Building vocabulary...", "vocab")
        self.init_vocab(merges)
        self.logger.log_complete("Vocabulary built", "vocab")

        # Log summary
        self.logger.log_training_summary("train", "word_counting", "merges", "vocab")

        return TrainResult(vocab=self.vocab, merges=merges)


# TODO: Remove this debug code when implementing full tokenizer
# This is currently here for testing the basic functionality
if __name__ == "__main__":
    bpe = BPETokenizer(GPT5_PAT_STR, MODEL_DEFAULTS.DEFAULT_SPECIAL_TOKENS, verbose=True)
    train_result = bpe.train(TINYSTORIES_VALID, 1024)
    print('vocab size = ', len(train_result.vocab))
    print('merges count = ', len(train_result.merges))
    print('First 10 merges:', train_result.merges[:10])

