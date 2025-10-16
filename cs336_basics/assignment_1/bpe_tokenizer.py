"""BPE tokenizer with memory-efficient training and deterministic merges.

Implements GPT-2/GPT-5 compatible byte-pair encoding with multiprocessing support.
Training uses chunked file processing (300MB chunks) to handle large datasets without
loading entire files into memory.

Algorithm: Iteratively merges most frequent byte pairs using lexicographic tie-breaking
for deterministic results. Special tokens are kept atomic and never split.

Classes:
    BPETokenizer: Main tokenizer with training capabilities
    TrainResult: Training output containing vocab and merge operations
"""

import regex as re
from collections import Counter, defaultdict
from dataclasses import dataclass
from cs336_basics.utils import (
    format_time,
    get_data_path,
    find_chunk_boundaries,
    process_file_chunks_multiprocessing,
    TrainingLogger,
    WordCountsCache,
)
from cs336_basics.constants import (
    GPT5_PAT_STR,
    END_OF_TEXT_TOKEN,
    MODEL_DEFAULTS,
    TINYSTORIES_VALID, TINYSTORIES_TRAIN, OWT_TRAIN
)


@dataclass
class TrainResult:
    """Result of BPE training."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


class BPETokenizer:
    """BPE (Byte Pair Encoding) Tokenizer for CS336 Assignment 1."""

    def __init__(self, pat_str: str, special_tokens: list[str], verbose: bool = False) -> None:
        """Initialize BPE tokenizer.

        Args:
            pat_str: Regex pattern for pre-tokenization (GPT2_PAT_STR or GPT5_PAT_STR)
            special_tokens: List of special tokens (e.g., ['<|endoftext|>'])
            verbose: Whether to print progress information
        """
        self.compiled_pat = re.compile(pat_str)
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: list[str] = special_tokens
        self.logger = TrainingLogger(verbose)
        self.cache = WordCountsCache(logger=self.logger)  # Pass logger to cache

        # Build combined pattern: special tokens | regular pattern
        if special_tokens:
            special_pattern = '|'.join(re.escape(token) for token in special_tokens)
            self.combined_pat = re.compile(f'({special_pattern})|{pat_str}')
        else:
            self.combined_pat = self.compiled_pat

    def init_vocab(self, merges: list[tuple[bytes, bytes]] | None = None) -> None:
        """Initialize vocabulary: special tokens + base bytes + merge results.

        Vocab IDs: [0..n-1]: special tokens, [n..n+255]: bytes, [n+256..]: merges
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
        """Count adjacent byte pairs weighted by word frequency."""
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for word, count in word_counts.items():
            # For each word, count all adjacent pairs
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count

        return pair_counts

    def _merge_pair(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """Merge all occurrences of a byte pair in a word."""
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
        """Apply merge to all words in Counter."""
        new_counts: Counter[tuple[bytes, ...]] = Counter()

        for word, count in word_counts.items():
            merged_word = self._merge_pair(word, pair)
            new_counts[merged_word] += count

        return new_counts

    def _process_chunk(self, file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
        """Process file chunk and return token counts."""
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

    def get_word_counts(self, filename: str, num_workers: int | None = None, use_cache: bool = True) -> Counter[tuple[bytes, ...]]:
        """Get word counts using multiprocessing with caching support."""
        self.logger.start_timer("word_counting")

        # Try cache first if enabled
        if use_cache:
            cached_counts = self.cache.load(filename)
            if cached_counts is not None:
                return cached_counts

        # Compute word counts from scratch
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

        # Save to cache if enabled
        if use_cache:
            self.cache.save(total_counts, filename, sort_by_frequency=True)

        return total_counts

    def train(self, input_path: str, vocab_size: int) -> TrainResult:
        """Train BPE tokenizer on input data."""
        self.logger.start_timer("train")
        self.logger.log_step("Step 1: Getting initial word counts...", "word_counting")

        word_counts = self.get_word_counts(input_path)
        self.logger.log_complete("Word counting", "word_counting")

        # Calculate number of merges needed
        num_special = len(self.special_tokens)
        num_merges_needed = vocab_size - num_special - 256
        self.logger.log_step(f"Step 2: Performing {num_merges_needed:,} BPE merges...", "merges")

        # Perform merges with incremental pair count updates
        merges: list[tuple[bytes, bytes]] = []

        # Initial pair counting (only done once)
        pair_counts = self._get_pair_counts(word_counts)

        for i in range(num_merges_needed):
            if not pair_counts:
                break

            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))

            # Track pair count changes using defaultdict for faster updates
            pair_deltas: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            new_word_counts: Counter[tuple[bytes, ...]] = Counter()

            # Apply merge and track changes
            for word, count in word_counts.items():
                # Fast check: skip words that are too short or don't need merging
                if len(word) < 2:
                    new_word_counts[word] += count
                    continue

                # Check if word actually contains the pair to merge
                has_pair = False
                for j in range(len(word) - 1):
                    if word[j] == best_pair[0][0] and word[j + 1] == best_pair[0][1]:
                        has_pair = True
                        break

                if has_pair:
                    # Get old pairs before merge
                    for j in range(len(word) - 1):
                        old_pair = (word[j], word[j + 1])
                        pair_deltas[old_pair] -= count

                    # Apply merge
                    merged_word = self._merge_pair(word, best_pair[0])
                    new_word_counts[merged_word] += count

                    # Get new pairs after merge
                    for j in range(len(merged_word) - 1):
                        new_pair = (merged_word[j], merged_word[j + 1])
                        pair_deltas[new_pair] += count
                else:
                    new_word_counts[word] += count

            word_counts = new_word_counts
            merges.append(best_pair[0])

            # Update pair_counts incrementally
            for pair, delta in pair_deltas.items():
                new_count = pair_counts.get(pair, 0) + delta
                if new_count > 0:
                    pair_counts[pair] = new_count
                elif pair in pair_counts:
                    del pair_counts[pair]

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
    train_result = bpe.train(TINYSTORIES_TRAIN, 4096)
    print('vocab size = ', len(train_result.vocab))
    print('merges count = ', len(train_result.merges))
    print('First 30 longest merges:', sorted(train_result.merges, key=lambda x: len(x[0] + x[1]), reverse=True)[:30])

