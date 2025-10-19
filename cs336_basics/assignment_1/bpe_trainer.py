"""BPE Trainer - Handles training orchestration for BPE tokenizer.

Separates training logic from the core tokenizer to follow Single Responsibility Principle.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from cs336_basics.utils import (
    format_time,
    get_data_path,
    find_chunk_boundaries,
    process_file_chunks_multiprocessing,
    TrainingLogger,
    WordCountsCache,
    PairCountsCache,
    MergesCache,
    VocabCache,
    CheckpointManager,
)
from cs336_basics.constants import END_OF_TEXT_TOKEN


@dataclass
class TrainingState:
    """Container for BPE training state."""
    word_counts: Counter[tuple[bytes, ...]]
    pair_counts: Counter[tuple[bytes, bytes]]
    merges: list[tuple[bytes, bytes]]
    start_iteration: int
    checkpoint_manager: CheckpointManager
    num_merges_needed: int


class BPETrainer:
    """Handles BPE tokenizer training with caching, checkpointing, and progress tracking."""

    def __init__(
        self,
        special_tokens: list[str],
        tokenization_pattern,
        verbose: bool = False
    ) -> None:
        """
        Initialize BPE trainer.

        Args:
            special_tokens: List of special tokens that should never be split
            tokenization_pattern: Compiled regex pattern for tokenization
            verbose: Whether to print progress information
        """
        self.special_tokens = special_tokens
        self.tokenization_pattern = tokenization_pattern
        self.logger = TrainingLogger(verbose)

        # Initialize caches
        self.word_cache = WordCountsCache(logger=self.logger)
        self.pair_cache = PairCountsCache(logger=self.logger)
        self.merges_cache = MergesCache(logger=self.logger)
        self.vocab_cache = VocabCache(logger=self.logger)

    def _process_chunk(self, file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
        """Process file chunk and return token counts."""
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8")

            word_counts: Counter[tuple[bytes, ...]] = Counter()

            # Use tokenization pattern to handle special tokens and regular text in one pass
            for match in self.tokenization_pattern.finditer(chunk_text):
                matched_text = match.group()

                # Check if this is a special token (matched by first capture group)
                if matched_text in self.special_tokens:
                    # Keep special token as a single token (atomic unit)
                    word = (matched_text.encode("utf-8"),)
                else:
                    # Keep regex-matched token as atomic unit, but split into individual bytes for BPE
                    # Each byte becomes a separate element in the tuple for BPE to merge
                    word = tuple(bytes([b]) for b in matched_text.encode("utf-8"))

                word_counts[word] += 1

            return word_counts

    def get_word_counts(
        self,
        filename: str,
        num_workers: int | None = None,
        use_cache: bool = True
    ) -> Counter[tuple[bytes, ...]]:
        """Get word counts using multiprocessing with caching support."""
        self.logger.start_timer("word_counting")

        # Try cache first if enabled
        if use_cache:
            cached_counts = self.word_cache.load(filename)
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
            self.word_cache.save(total_counts, filename)

        return total_counts

    def _get_pair_counts(self, word_counts: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
        """Count adjacent byte pairs weighted by word frequency."""
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for word, count in word_counts.items():
            # For each word, count all adjacent pairs
            for position in range(len(word) - 1):
                pair = (word[position], word[position + 1])
                pair_counts[pair] += count

        return pair_counts

    def _merge_pair(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """Merge all occurrences of a byte pair in a word."""
        if len(word) < 2:
            return word

        merged_sequence = []
        position = 0

        while position < len(word):
            # If we found the pair and it's not the last byte, merge it
            if position < len(word) - 1 and (word[position], word[position + 1]) == pair:
                # Merge the two bytes
                merged_sequence.append(word[position] + word[position + 1])
                position += 2
            else:
                merged_sequence.append(word[position])
                position += 1

        return tuple(merged_sequence)

    def initialize_training_state(
        self,
        input_path: str,
        vocab_size: int
    ) -> TrainingState:
        """
        Initialize training state with word counts, pair counts, merges, and checkpoint data.

        Args:
            input_path: Path to training data file
            vocab_size: Target vocabulary size

        Returns:
            TrainingState with all necessary data for training
        """
        # Step 1: Get word counts
        self.logger.log_step("Step 1: Getting initial word counts...", "word_counting")
        word_counts = self.get_word_counts(input_path)
        self.logger.log_complete("Word counting", "word_counting")

        # Calculate number of merges needed
        num_special = len(self.special_tokens)
        num_merges_needed = vocab_size - num_special - 256
        self.logger.log_step(f"Step 2: Performing {num_merges_needed:,} BPE merges...")

        # Step 2: Get initial pair counts (with caching)
        self.logger.start_timer("pair_counting")
        cached_pair_counts = self.pair_cache.load(input_path)

        if cached_pair_counts is not None:
            pair_counts = cached_pair_counts
        else:
            # Cache miss - compute from word_counts
            pair_counts = self._get_pair_counts(word_counts)
            # Save to cache for future runs
            self.pair_cache.save(pair_counts, input_path)

        # Step 3: Load checkpoint if available
        dataset_name = CheckpointManager.get_dataset_name(input_path)
        checkpoint_manager = CheckpointManager(dataset_name, logger=self.logger)

        checkpoint_state = checkpoint_manager.load_or_initialize(num_merges_needed, self.special_tokens)
        merges = checkpoint_state['merges']
        start_iteration = checkpoint_state['start_iteration']

        # Start merges timer with correct offset for resumable operations
        self.logger.start_timer("merges", start_iteration)

        # Override word_counts and pair_counts if loaded from checkpoint
        if checkpoint_state['word_counts'] is not None:
            word_counts = checkpoint_state['word_counts']
            pair_counts = checkpoint_state['pair_counts']

        return TrainingState(
            word_counts=word_counts,
            pair_counts=pair_counts,
            merges=merges,
            start_iteration=start_iteration,
            checkpoint_manager=checkpoint_manager,
            num_merges_needed=num_merges_needed
        )

    def run_merge_loop(
        self,
        state: TrainingState,
        checkpoint_interval: int = 300
    ) -> list[tuple[bytes, bytes]]:
        """
        Execute the BPE merge loop.

        Args:
            state: Training state containing word counts, pair counts, etc.
            checkpoint_interval: How often to save checkpoints (default: every 300 merges)

        Returns:
            List of merge operations performed
        """
        word_counts = state.word_counts
        pair_counts = state.pair_counts
        merges = state.merges

        for merge_iteration in range(state.start_iteration, state.num_merges_needed):
            if not pair_counts:
                break

            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
            pair_to_merge = most_frequent_pair[0]

            # Apply merge to all words
            pair_deltas: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            new_word_counts: Counter[tuple[bytes, ...]] = Counter()

            for word, count in word_counts.items():
                if len(word) < 2:
                    new_word_counts[word] += count
                    continue

                # Check if word actually contains the pair to merge
                has_pair = False
                for pos in range(len(word) - 1):
                    if word[pos] == pair_to_merge[0] and word[pos + 1] == pair_to_merge[1]:
                        has_pair = True
                        break

                if has_pair:
                    # Get old pairs before merge
                    for pos in range(len(word) - 1):
                        old_pair = (word[pos], word[pos + 1])
                        pair_deltas[old_pair] -= count

                    # Apply merge
                    merged_word = self._merge_pair(word, pair_to_merge)
                    new_word_counts[merged_word] += count

                    # Get new pairs after merge
                    for pos in range(len(merged_word) - 1):
                        new_pair = (merged_word[pos], merged_word[pos + 1])
                        pair_deltas[new_pair] += count
                else:
                    new_word_counts[word] += count

            word_counts = new_word_counts
            merges.append(pair_to_merge)

            # Update pair_counts incrementally
            for pair, delta in pair_deltas.items():
                new_count = pair_counts.get(pair, 0) + delta
                if new_count > 0:
                    pair_counts[pair] = new_count
                elif pair in pair_counts:
                    del pair_counts[pair]

            # Log progress adaptively (every 10% or 10 minutes)
            self.logger.log_progress_adaptive(
                merge_iteration + 1,
                state.num_merges_needed,
                "merges",
                progress_interval_seconds=10 * 60
            )

            # Save checkpoint every N merges
            if (merge_iteration + 1) % checkpoint_interval == 0:
                state.checkpoint_manager.save_checkpoint_from_state(
                    iteration=merge_iteration + 1,
                    merges=merges,
                    word_counts=word_counts,
                    pair_counts=pair_counts,
                    special_tokens=self.special_tokens
                )

        self.logger.log_complete("BPE merges", "merges")
        return merges