"""
Training logger for tracking progress during BPE training and other operations.

This module provides a clean interface for logging training progress without
cluttering the main training logic with print statements.
"""

import time
from typing import Any
from cs336_basics.utils import format_time


class TrainingLogger:
    """Logger for tracking and displaying training progress."""

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the training logger.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.timers: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.time()

    def get_elapsed(self, name: str) -> float:
        """Get elapsed time for a named timer."""
        if name not in self.timers:
            return 0.0
        return time.time() - self.timers[name]

    def log(self, message: str, **kwargs: Any) -> None:
        """
        Log a message if verbose mode is enabled.

        Args:
            message: Message to log
            **kwargs: Optional formatting arguments
        """
        if self.verbose:
            if kwargs:
                print(message.format(**kwargs))
            else:
                print(message)

    def log_step(self, step_name: str, timer_name: str | None = None) -> None:
        """
        Log the start of a training step and optionally start a timer.

        Args:
            step_name: Name of the step to display
            timer_name: Optional timer name to start (defaults to step_name if not provided)
        """
        if self.verbose:
            print(f"\n{step_name}")
        if timer_name is not None:
            self.start_timer(timer_name)
        elif timer_name is None and self.verbose:
            # Auto-derive timer name from step name if verbose
            # e.g., "Step 1: Getting word counts" -> "word_counts"
            pass  # Don't auto-start timer unless explicitly requested

    def log_complete(self, step_name: str, timer_name: str) -> None:
        """Log completion of a step with elapsed time."""
        if self.verbose:
            elapsed = self.get_elapsed(timer_name)
            print(f"✓ {step_name} completed in {format_time(elapsed)}")

    def log_progress(self, current: int, total: int, timer_name: str) -> None:
        """
        Log progress during an iterative operation.

        Args:
            current: Current iteration number
            total: Total number of iterations
            timer_name: Name of the timer tracking this operation
        """
        if self.verbose:
            progress = (current / total) * 100
            elapsed = self.get_elapsed(timer_name)
            print(f"  Progress: {current:,}/{total:,} ({progress:.0f}%) - {format_time(elapsed)} elapsed")

    def log_summary(self, title: str, items: dict[str, tuple[float, float]]) -> None:
        """
        Log a summary table with timing breakdown.

        Args:
            title: Title for the summary
            items: Dict mapping item names to (time, percentage) tuples
        """
        if not self.verbose:
            return

        print(f"\n{'='*60}")
        print(title)
        for name, (elapsed, percentage) in items.items():
            print(f"  - {name}: {format_time(elapsed)} ({percentage:.1f}%)")
        print(f"{'='*60}")

    def log_training_summary(self, main_timer: str, *step_timers: str) -> None:
        """
        Log a training summary with automatic time calculation.

        Args:
            main_timer: Name of the main timer (e.g., "train")
            *step_timers: Names of step timers to include in breakdown
        """
        if not self.verbose:
            return

        total_time = self.get_elapsed(main_timer)

        items = {}
        for timer_name in step_timers:
            step_time = self.get_elapsed(timer_name)
            # Convert timer name to display name (e.g., "word_counting" -> "Word counting")
            display_name = timer_name.replace("_", " ").capitalize()
            items[display_name] = (step_time, step_time / total_time * 100 if total_time > 0 else 0)

        self.log_summary(f"Total training time: {format_time(total_time)}", items)

    def log_stats(self, **stats: Any) -> None:
        """
        Log statistics as key-value pairs.

        Args:
            **stats: Statistics to log
        """
        if not self.verbose:
            return

        for key, value in stats.items():
            if isinstance(value, int):
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")