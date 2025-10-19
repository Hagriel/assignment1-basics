"""Checkpoint cache for resumable BPE training.

Saves incremental checkpoints every N merges that can be reused across different vocab sizes.
Structure: data/cache/{dataset}/vocab_{iteration}.json
                                /merges_{iteration}.json
                                /word_counts_{iteration}.json
                                /pair_counts_{iteration}.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
from .base import JSONCache
from .vocab import VocabCache
from .merges import MergesCache
from .word_counts import WordCountsCache
from .pair_counts import PairCountsCache


class CheckpointManager:
    """Manages incremental BPE training checkpoints for resumable training."""

    def __init__(self, dataset_name: str, logger: Any | None = None) -> None:
        """
        Initialize checkpoint manager.

        Args:
            dataset_name: Name of dataset (used for checkpoint directory)
            logger: Optional logger for verbose output
        """
        self.dataset_name = dataset_name
        self.logger = logger

        # Create checkpoint directory (same as other cache files)
        from cs336_basics.utils import get_project_root
        base_cache_dir = get_project_root() / "data" / "cache"
        self.checkpoint_dir = base_cache_dir / dataset_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use existing cache classes for serialization
        self.vocab_cache = VocabCache(logger=logger)
        self.merges_cache = MergesCache(logger=logger)
        self.word_counts_cache = WordCountsCache(logger=logger)
        self.pair_counts_cache = PairCountsCache(logger=logger)

    @staticmethod
    def get_dataset_name(input_path: str) -> str:
        """Extract dataset name from input path."""
        if '/' in input_path or '\\' in input_path:
            return Path(input_path).stem
        return input_path.replace('.txt', '')

    @staticmethod
    def build_vocab(special_tokens: list[str], merges: list[tuple[bytes, bytes]]) -> dict[int, bytes]:
        """Build vocabulary from special tokens and merges."""
        vocab = {}

        # Special tokens
        for token_idx, token in enumerate(special_tokens):
            vocab[token_idx] = token.encode('utf-8')

        num_special = len(special_tokens)

        # Base bytes
        for byte_value in range(256):
            vocab[num_special + byte_value] = bytes([byte_value])

        # Merges
        for merge_idx, (left, right) in enumerate(merges):
            vocab[num_special + 256 + merge_idx] = left + right

        return vocab

    def load_or_initialize(self, num_merges_needed: int, special_tokens: list[str]) -> dict[str, Any]:
        """
        Load nearest checkpoint or return initial state.

        Args:
            num_merges_needed: Target number of merges
            special_tokens: List of special tokens

        Returns:
            Dictionary with 'merges', 'word_counts', 'pair_counts', 'vocab', 'start_iteration'
        """
        nearest_checkpoint_iter = self.find_nearest_checkpoint(num_merges_needed)

        if nearest_checkpoint_iter is not None:
            checkpoint = self.load_checkpoint(nearest_checkpoint_iter)
            if checkpoint is not None:
                if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
                    print(f"Resuming from checkpoint {nearest_checkpoint_iter:,}/{num_merges_needed:,} merges")
                return {
                    'merges': checkpoint['merges'],
                    'word_counts': checkpoint['word_counts'],
                    'pair_counts': checkpoint['pair_counts'],
                    'vocab': checkpoint['vocab'],
                    'start_iteration': checkpoint['iteration']
                }

        # No checkpoint found - return initial state
        initial_vocab = self.build_vocab(special_tokens, [])
        return {
            'merges': [],
            'word_counts': None,
            'pair_counts': None,
            'vocab': initial_vocab,
            'start_iteration': 0
        }

    def save_checkpoint_from_state(
        self,
        iteration: int,
        merges: list[tuple[bytes, bytes]],
        word_counts: Counter[tuple[bytes, ...]],
        pair_counts: Counter[tuple[bytes, bytes]],
        special_tokens: list[str]
    ) -> None:
        """
        Save checkpoint, building vocab from current state.

        Args:
            iteration: Merge iteration number
            merges: List of merges up to this iteration
            word_counts: Current word counts
            pair_counts: Current pair counts
            special_tokens: List of special tokens
        """
        # Build vocab from current merges
        current_vocab = self.build_vocab(special_tokens, merges)

        # Save checkpoint
        self.save_checkpoint(
            iteration=iteration,
            merges=merges,
            word_counts=word_counts,
            pair_counts=pair_counts,
            vocab=current_vocab
        )

    def save_checkpoint(
        self,
        iteration: int,
        merges: list[tuple[bytes, bytes]],
        word_counts: Counter[tuple[bytes, ...]],
        pair_counts: Counter[tuple[bytes, bytes]],
        vocab: dict[int, bytes]
    ) -> None:
        """
        Save checkpoint at given iteration.

        Args:
            iteration: Merge iteration number
            merges: List of merges up to this iteration
            word_counts: Current word counts
            pair_counts: Current pair counts
            vocab: Current vocabulary
        """
        # Save each component separately
        vocab_path = self.checkpoint_dir / f"vocab_{iteration}.json"
        merges_path = self.checkpoint_dir / f"merges_{iteration}.json"
        word_counts_path = self.checkpoint_dir / f"word_counts_{iteration}.json"
        pair_counts_path = self.checkpoint_dir / f"pair_counts_{iteration}.json"

        # Serialize using existing cache classes
        import json

        # Save vocab
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab_cache.serialize(vocab), f, indent=2)

        # Save merges
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(self.merges_cache.serialize(merges), f, indent=2)

        # Save word_counts
        with open(word_counts_path, 'w', encoding='utf-8') as f:
            json.dump(self.word_counts_cache.serialize(word_counts), f, indent=2)

        # Save pair_counts
        with open(pair_counts_path, 'w', encoding='utf-8') as f:
            json.dump(self.pair_counts_cache.serialize(pair_counts), f, indent=2)

        if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
            print(f"Checkpoint saved: iteration {iteration:,}")

    def find_nearest_checkpoint(self, target_iteration: int) -> int | None:
        """
        Find the nearest checkpoint <= target_iteration.

        Args:
            target_iteration: Target merge iteration

        Returns:
            Iteration number of nearest checkpoint, or None if no checkpoint exists
        """
        # Find all vocab checkpoint files
        vocab_checkpoints = list(self.checkpoint_dir.glob("vocab_*.json"))

        if not vocab_checkpoints:
            return None

        # Extract iteration numbers
        iterations = []
        for path in vocab_checkpoints:
            try:
                iteration = int(path.stem.split('_')[1])
                if iteration <= target_iteration:
                    iterations.append(iteration)
            except (ValueError, IndexError):
                continue

        return max(iterations) if iterations else None

    def load_checkpoint(self, iteration: int) -> dict[str, Any] | None:
        """
        Load checkpoint at given iteration.

        Args:
            iteration: Merge iteration number

        Returns:
            Dictionary with 'iteration', 'merges', 'word_counts', 'pair_counts', 'vocab'
            or None if checkpoint doesn't exist
        """
        vocab_path = self.checkpoint_dir / f"vocab_{iteration}.json"
        merges_path = self.checkpoint_dir / f"merges_{iteration}.json"
        word_counts_path = self.checkpoint_dir / f"word_counts_{iteration}.json"
        pair_counts_path = self.checkpoint_dir / f"pair_counts_{iteration}.json"

        # Check all files exist
        if not all([vocab_path.exists(), merges_path.exists(),
                    word_counts_path.exists(), pair_counts_path.exists()]):
            return None

        try:
            import json

            # Load vocab
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = self.vocab_cache.deserialize(json.load(f))

            # Load merges
            with open(merges_path, 'r', encoding='utf-8') as f:
                merges = self.merges_cache.deserialize(json.load(f))

            # Load word_counts
            with open(word_counts_path, 'r', encoding='utf-8') as f:
                word_counts = self.word_counts_cache.deserialize(json.load(f))

            # Load pair_counts
            with open(pair_counts_path, 'r', encoding='utf-8') as f:
                pair_counts = self.pair_counts_cache.deserialize(json.load(f))

            if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
                print(f"Checkpoint loaded: iteration {iteration:,}")

            return {
                'iteration': iteration,
                'merges': merges,
                'word_counts': word_counts,
                'pair_counts': pair_counts,
                'vocab': vocab
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
                print(f"Failed to load checkpoint at iteration {iteration}: {e}")
            return None

    def list_checkpoints(self) -> list[int]:
        """
        List all available checkpoint iterations.

        Returns:
            Sorted list of iteration numbers
        """
        vocab_checkpoints = list(self.checkpoint_dir.glob("vocab_*.json"))
        iterations = []

        for path in vocab_checkpoints:
            try:
                iteration = int(path.stem.split('_')[1])
                iterations.append(iteration)
            except (ValueError, IndexError):
                continue

        return sorted(iterations)

    def delete_checkpoint(self, iteration: int) -> None:
        """
        Delete checkpoint at given iteration.

        Args:
            iteration: Merge iteration number
        """
        files_to_delete = [
            self.checkpoint_dir / f"vocab_{iteration}.json",
            self.checkpoint_dir / f"merges_{iteration}.json",
            self.checkpoint_dir / f"word_counts_{iteration}.json",
            self.checkpoint_dir / f"pair_counts_{iteration}.json"
        ]

        for file_path in files_to_delete:
            if file_path.exists():
                file_path.unlink()

        if self.logger and hasattr(self.logger, 'verbose') and self.logger.verbose:
            print(f"Checkpoint deleted: iteration {iteration:,}")

