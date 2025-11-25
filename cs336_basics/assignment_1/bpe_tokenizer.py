"""BPE tokenizer with memory-efficient training and deterministic merges.

Implements GPT-2/GPT-5 compatible byte-pair encoding with multiprocessing support.
Training uses chunked file processing (300MB chunks) to handle large datasets without
loading entire files into memory.

Algorithm: Iteratively merges most frequent byte pairs using lexicographic tie-breaking
for deterministic results. Special tokens are kept atomic and never split.

Classes:
    BPETokenizer: Extends Tokenizer with BPE training functionality
    TrainResult: Training output containing vocab and merge operations
    TrainingState: Container for BPE training state
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from cs336_basics.assignment_1.tokenizer import Tokenizer
from cs336_basics.utils import (
    format_time,
    get_data_path,
    find_chunk_boundaries,
    process_file_chunks_multiprocessing,
    TrainingLogger,
    WordCountsCache,
    PairCountsCache,
    CheckpointManager,
)
from cs336_basics.constants import (
    END_OF_TEXT_TOKEN,
    GPT5_PAT_STR,
    MODEL_DEFAULTS,
    TINYSTORIES_VALID,
    TINYSTORIES_TRAIN,
    OWT_TRAIN,
    GPT2_PAT_STR,
)


@dataclass
class TrainResult:
    """Result of BPE training."""
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


@dataclass
class TrainingState:
    """Container for BPE training state."""
    word_counts: Counter[tuple[bytes, ...]]
    pair_counts: Counter[tuple[bytes, bytes]]
    merges: list[tuple[bytes, bytes]]
    start_iteration: int
    checkpoint_manager: CheckpointManager
    num_merges_needed: int
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] | None = None


class BPETokenizer(Tokenizer):
    """BPE (Byte Pair Encoding) Tokenizer with training functionality.

    Extends the base Tokenizer class with BPE training capabilities.
    """

    def __init__(self, pattern: str, special_tokens: list[str], verbose: bool = False) -> None:
        """Initialize BPE tokenizer.

        Args:
            pattern: Regex pattern for pre-tokenization (GPT2_PAT_STR or GPT5_PAT_STR)
            special_tokens: List of special tokens (e.g., ['<|endoftext|>'])
            verbose: Whether to print progress information
        """
        # Initialize base tokenizer (handles vocab, encoding, decoding)
        super().__init__(pattern, special_tokens, verbose)

        # Initialize training-specific components
        self.logger = TrainingLogger(verbose)
        self.word_cache = WordCountsCache(logger=self.logger)
        self.pair_cache = PairCountsCache(logger=self.logger)

    def _process_chunk(self, file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
        """Process file chunk and return token counts."""
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8")

            word_counts: Counter[tuple[bytes, ...]] = Counter()

            for doc in self.special_tokens_pattern.split(chunk_text):
                # Use tokenization pattern to handle special tokens and regular text in one pass
                for match in self.pretokenization_pattern.finditer(doc):
                    matched_text = match.group()

                    # Check if this is a special token (matched by first capture group)
                    if matched_text in self.special_tokens:
                        continue
                    else:
                        # Keep regex-matched token as atomic unit, but split into individual bytes for BPE
                        # Each byte becomes a separate element in the tuple for BPE to merge
                        word = tuple(bytes([b]) for b in matched_text.encode("utf-8"))

                    word_counts[word] += 1

            return word_counts

    def _get_word_counts(
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
                target_chunk_size= 300 * 1024 * 1024
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

    def _build_pair_index(self, word_counts: Counter[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
        """Build reverse index: pair -> set of words containing that pair.

        This enables O(words_with_pair) lookup instead of O(all_words) for each merge.
        Memory trade-off: ~1-2GB extra for large datasets, but 50-100x faster merges.

        Args:
            word_counts: Current word counts

        Returns:
            Dictionary mapping each pair to the set of words containing it
        """
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

        for word in word_counts:
            if len(word) < 2:
                continue
            # Add word to index for each pair it contains
            for position in range(len(word) - 1):
                pair = (word[position], word[position + 1])
                pair_to_words[pair].add(word)

        return pair_to_words

    def _initialize_training_state(
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
        word_counts = self._get_word_counts(input_path)
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

        # Build pair-to-words index for fast lookup (trades memory for speed)
        self.logger.log_step("Building pair-to-words index for fast merges...")
        pair_to_words = self._build_pair_index(word_counts)
        self.logger.log_stats(**{
            "Index built": "",
            "Pairs indexed": len(pair_to_words),
            "Average words per pair": sum(len(words) for words in pair_to_words.values()) / len(pair_to_words) if pair_to_words else 0
        })

        return TrainingState(
            word_counts=word_counts,
            pair_counts=pair_counts,
            merges=merges,
            start_iteration=start_iteration,
            checkpoint_manager=checkpoint_manager,
            num_merges_needed=num_merges_needed,
            pair_to_words=pair_to_words
        )

    def _run_merge_loop(
        self,
        state: TrainingState,
        checkpoint_interval: int = 300
    ) -> list[tuple[bytes, bytes]]:
        """
        Execute BPE merge loop with index-based lookup.

        Args:
            state: Training state containing word counts, pair counts, and index
            checkpoint_interval: How often to save checkpoints (default: every 300 merges)

        Returns:
            List of merge operations performed
        """
        word_counts = state.word_counts
        pair_counts = state.pair_counts
        merges = state.merges
        pair_to_words = state.pair_to_words

        if pair_to_words is None:
            raise ValueError("Index-based merge loop requires pair_to_words index")

        for merge_iteration in range(state.start_iteration, state.num_merges_needed):
            if not pair_counts:
                break

            # Find most frequent pair (sort by count desc, then by pair lexicographically for deterministic tie-breaking)
            most_frequent_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            pair_to_merge = most_frequent_pair[0]

            # Get only words containing this specific pair (KEY OPTIMIZATION!)
            affected_words = pair_to_words.get(pair_to_merge, set()).copy()

            if not affected_words:
                # Pair no longer exists in any word - skip it
                del pair_counts[pair_to_merge]
                continue

            # Track pair count changes
            pair_deltas: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)

            # Process only words containing the pair to merge
            for word in affected_words:
                count = word_counts.get(word, 0)
                if count == 0:
                    continue  # Word was already merged away

                # Record old pairs before merge
                for position in range(len(word) - 1):
                    old_pair = (word[position], word[position + 1])
                    pair_deltas[old_pair] -= count
                    # Remove word from old pair's index
                    if old_pair in pair_to_words:
                        pair_to_words[old_pair].discard(word)

                # Apply merge
                merged_word = self._merge_pair(word, pair_to_merge)

                # Update word counts
                del word_counts[word]
                word_counts[merged_word] += count

                # Record new pairs after merge
                for position in range(len(merged_word) - 1):
                    new_pair = (merged_word[position], merged_word[position + 1])
                    pair_deltas[new_pair] += count
                    # Add merged word to new pair's index
                    pair_to_words[new_pair].add(merged_word)

            merges.append(pair_to_merge)

            # Clean up the merged pair from index
            if pair_to_merge in pair_to_words:
                del pair_to_words[pair_to_merge]

            # Update pair_counts incrementally
            for pair, delta in pair_deltas.items():
                new_count = pair_counts.get(pair, 0) + delta
                if new_count > 0:
                    pair_counts[pair] = new_count
                else:
                    # Remove pair if count drops to 0
                    if pair in pair_counts:
                        del pair_counts[pair]
                    # Clean up empty entries in index
                    if pair in pair_to_words and not pair_to_words[pair]:
                        del pair_to_words[pair]

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

    def train(self, input_path: str, vocab_size: int) -> TrainResult:
        """Train BPE tokenizer on input data.

        Args:
            input_path: Path to training data file
            vocab_size: Target vocabulary size

        Returns:
            TrainResult with vocabulary and merge operations
        """
        # Try to load from cache first using data manager
        vocab_state = self.data_manager.try_load_vocab(
            special_tokens=self.special_tokens,
            dataset_name=input_path,
            vocab_size=vocab_size
        )

        if vocab_state is not None:
            # Cache hit - use loaded state
            self.vocab = vocab_state.vocab
            self.merges = vocab_state.merges
            self._bytes_to_id = vocab_state.bytes_to_id
            return TrainResult(vocab=self.vocab, merges=self.merges)

        # Cache miss - run full training
        # Initialize training state
        state = self._initialize_training_state(input_path, vocab_size)

        merges = self._run_merge_loop(state, checkpoint_interval=1000)

        # Build vocabulary
        self.logger.log_step("Step 3: Building vocabulary...", "vocab")
        self.init_vocab(merges=merges)
        self.merges = merges
        self.logger.log_complete("Vocabulary built", "vocab")

        # Save to cache using data manager
        self.data_manager.save_to_cache(
            dataset_name=input_path,
            vocab_size=vocab_size,
            vocab=self.vocab,
            merges=self.merges
        )

        # Log summary
        self.logger.log_training_summary("train", "word_counting", "merges", "vocab")

        return TrainResult(vocab=self.vocab, merges=merges)