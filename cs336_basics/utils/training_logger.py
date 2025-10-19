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
        # Track last progress state for calculating block speed
        self.last_progress: dict[str, dict[str, float]] = {}  # {timer_name: {count: int, time: float}}
        # Track starting offset for each timer (for resumable operations)
        self.start_offsets: dict[str, int] = {}  # {timer_name: starting_count}
        # Track last progress log time for adaptive intervals
        self.last_progress_log_time: dict[str, float] = {}  # {timer_name: last_log_time}

    def start_timer(self, name: str, start_count: int = 0) -> None:
        """
        Start a named timer.

        Args:
            name: Name of the timer
            start_count: Starting iteration count (for resumable operations)
        """
        self.timers[name] = time.time()
        self.start_offsets[name] = start_count

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
        Log progress during an iterative operation with average and current block speed.

        Args:
            current: Current iteration number (absolute value)
            total: Total number of iterations
            timer_name: Name of the timer tracking this operation
        """
        if self.verbose:
            # Track starting offset on first call (for resumable operations)
            if timer_name not in self.start_offsets:
                self.start_offsets[timer_name] = current

            # Calculate incremental progress since timer started
            start_offset = self.start_offsets[timer_name]
            incremental_progress = current - start_offset

            progress = (current / total) * 100
            elapsed = self.get_elapsed(timer_name)

            # Calculate average speed (based on incremental progress)
            avg_speed = incremental_progress / elapsed if elapsed > 0 else 0

            # Calculate current block speed (since last progress log)
            if timer_name in self.last_progress:
                last_count = self.last_progress[timer_name]['count']
                last_time = self.last_progress[timer_name]['time']
                block_count = current - last_count
                block_time = elapsed - last_time
                current_speed = block_count / block_time if block_time > 0 else 0
            else:
                # First progress log - current speed equals average speed
                current_speed = avg_speed

            # Update tracking for next progress log
            self.last_progress[timer_name] = {'count': current, 'time': elapsed}

            # Calculate remaining time based on average speed
            remaining = (total - current) / avg_speed if avg_speed > 0 else 0

            # Format speed strings
            avg_speed_str = f"{avg_speed:.2f} per sec"
            current_speed_str = f"{current_speed:.2f} per sec"
            remaining_str = f"~{format_time(remaining)} remaining"

            print(f"  Progress: {current:,}/{total:,} ({progress:.0f}%) - {format_time(elapsed)} elapsed (avg: {avg_speed_str}, current: {current_speed_str}, {remaining_str})")

    def log_progress_adaptive(
        self,
        current: int,
        total: int,
        timer_name: str,
        progress_interval_seconds: int = 600
    ) -> None:
        """
        Log progress adaptively: every 10% milestone OR every N seconds (whichever comes first).

        Args:
            current: Current iteration number (absolute value)
            total: Total number of iterations
            timer_name: Name of the timer tracking this operation
            progress_interval_seconds: Time interval in seconds (default: 600 = 10 minutes)
        """
        if not self.verbose:
            return

        # Initialize last log time on first call
        if timer_name not in self.last_progress_log_time:
            self.last_progress_log_time[timer_name] = time.time()

        current_time = time.time()
        time_since_last_log = current_time - self.last_progress_log_time[timer_name]

        # Calculate if we're at a 10% milestone
        is_10_percent_milestone = (current % max(1, total // 10)) == 0

        # Check if time interval elapsed
        is_time_interval_elapsed = time_since_last_log >= progress_interval_seconds

        # Log if either condition met
        if is_10_percent_milestone or is_time_interval_elapsed:
            self.log_progress(current, total, timer_name)
            self.last_progress_log_time[timer_name] = current_time


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